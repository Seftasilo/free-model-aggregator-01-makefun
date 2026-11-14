// The Gemini wire, both directions.
//
// A Gemini provider speaks the vendor's own API, not the OpenAI or Anthropic one:
//   POST <base>/v1beta/models/<model>:generateContent        (a JSON answer)
//   POST <base>/v1beta/models/<model>:streamGenerateContent  (server-sent events, ?alt=sse accepted)
//   GET  <base>/v1beta/models                                (the catalog probe, in lib/providers.ts)
//
// A request body carries the system prompt as systemInstruction, the conversation as
// contents with user/model roles, the tools as functionDeclarations, and the output
// controls as generationConfig. An answer carries candidates with content parts, a
// finishReason and usageMetadata. Nothing here reads a clock or rolls dice, so the
// same request and the same provider answer always give the same bytes back.

import { isRecord, type Json } from './providers.ts';

/** One Gemini content part: text, an inline image, a file image, or a function call. */
export type GeminiPart = Json;

/** A fixed epoch second for every OpenAI-shaped answer's `created`: nothing reads the clock. */
const FIXED_CREATED = 1767225600;

/** Gemini's finishReason as an OpenAI finish_reason. */
export function geminiStopToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  if (reason === 'STOP') return 'stop';
  return 'stop';
}

/** Gemini's finishReason as an Anthropic stop_reason. */
export function geminiStopToAnthropic(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  return 'end_turn';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The text of a string, or of a block array's text blocks. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The text parts of one Gemini content's parts array, joined as the provider joins them. */
function geminiText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('\n');
}

/** One OpenAI content part as a Gemini part: text, or an image as inline data / a file URI. */
function geminiPartOfOpenAI(part: Json): GeminiPart | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const dataUrl = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
    if (dataUrl !== null) return { inlineData: { mimeType: dataUrl[1] ?? 'image/png', data: dataUrl[2] } };
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** One Anthropic content block as a Gemini part: text, image, or a tool use / result. */
function geminiPartOfAnthropic(block: Json): GeminiPart | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  if (block.type === 'tool_use') {
    return { functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } };
  }
  if (block.type === 'tool_result') {
    return {
      functionResponse: {
        name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
        response: { content: typeof block.content === 'string' ? block.content : textOf(block.content) },
      },
    };
  }
  return null;
}

/** An OpenAI function tool as a Gemini function declaration. */
function declarationOfOpenAITool(tool: Json): Json | null {
  if (!isRecord(tool.function) || typeof tool.function.name !== 'string') return null;
  return {
    name: tool.function.name,
    description: typeof tool.function.description === 'string' ? tool.function.description : '',
    parameters: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
  };
}

/** An Anthropic tool as a Gemini function declaration. */
function declarationOfAnthropicTool(tool: Json): Json | null {
  if (typeof tool.name !== 'string') return null;
  return {
    name: tool.name,
    description: typeof tool.description === 'string' ? tool.description : '',
    parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
  };
}

/** The generationConfig for a request: output cap, sampling, stops, and JSON mode. */
function generationConfigOf(source: Json, jsonMode: boolean): Json {
  const config: Json = {};
  const maxTokens =
    typeof source.max_tokens === 'number'
      ? source.max_tokens
      : typeof source.max_completion_tokens === 'number'
        ? source.max_completion_tokens
        : typeof source.maxTokens === 'number'
          ? source.maxTokens
          : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof source.temperature === 'number') config.temperature = source.temperature;
  if (typeof source.top_p === 'number') config.topP = source.top_p;
  const stops = Array.isArray(source.stop) ? source.stop : Array.isArray(source.stop_sequences) ? source.stop_sequences : [];
  const sequences = (stops as unknown[]).filter((s): s is string => typeof s === 'string');
  if (sequences.length > 0) config.stopSequences = sequences;
  if (jsonMode) config.responseMimeType = 'application/json';
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body naming `model`.
 * The model id travels in the URL, not the body. System messages become the
 * systemInstruction; user and assistant messages become contents; tool messages
 * become user contents carrying the result text, so the provider echoes the same
 * user text the client sent.
 */
export function geminiRequestFromOpenAI(body: Json, model: string): { body: Json; model: string } {
  const contents: Json[] = [];
  const systemParts: GeminiPart[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push({ text });
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: GeminiPart[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let args: Json = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) args = parsed;
          } catch {
            args = {};
          }
        }
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts: GeminiPart[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        const part = geminiPartOfOpenAI(block);
        if (part !== null) parts.push(part);
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((t) => {
      if (!isRecord(t)) return [];
      const d = declarationOfOpenAITool(t);
      return d === null ? [] : [d];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const config = generationConfigOf(body, format === 'json_object' || format === 'json_schema');
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return { body: out, model };
}

/**
 * An Anthropic Messages request as a Gemini generateContent body naming `model`.
 * The system prompt becomes the systemInstruction, user blocks become user contents
 * (text, images and tool results), assistant blocks become model contents (text and
 * tool uses), and the Anthropic tools become function declarations.
 */
export function geminiRequestFromAnthropic(body: Json, model: string): { body: Json; model: string } {
  const contents: Json[] = [];
  const systemText = textOf(body.system);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const parts: GeminiPart[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        const part = geminiPartOfAnthropic(block);
        if (part !== null) parts.push(part);
      }
    }
    if (parts.length === 0) continue;
    contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((t) => {
      if (!isRecord(t)) return [];
      const d = declarationOfAnthropicTool(t);
      return d === null ? [] : [d];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config = generationConfigOf(body, false);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return { body: out, model };
}

/** The first candidate of a Gemini answer: its content, finish reason and usage. */
function firstCandidate(answer: Json): { parts: unknown; finishReason: unknown; usage: Json } {
  const candidates = Array.isArray(answer.candidates) ? answer.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const usage = isRecord(answer.usageMetadata) ? answer.usageMetadata : {};
  return { parts: content.parts, finishReason: candidate.finishReason, usage };
}

/** A Gemini answer's text, with any function calls the model made. */
function answerTextAndCalls(parts: unknown): { text: string; calls: { name: string; args: Json }[] } {
  const calls: { name: string; args: Json }[] = [];
  const texts: string[] = [];
  if (Array.isArray(parts)) {
    for (const part of parts) {
      if (!isRecord(part)) continue;
      if (typeof part.text === 'string') texts.push(part.text);
      if (isRecord(part.functionCall) && typeof part.functionCall.name === 'string') {
        calls.push({ name: part.functionCall.name, args: isRecord(part.functionCall.args) ? part.functionCall.args : {} });
      }
    }
  }
  return { text: texts.join('\n'), calls };
}

/** A Gemini JSON answer as an OpenAI Chat Completions object naming `model`. */
export function geminiToOpenAI(answer: Json, model: string): Json {
  const { parts, finishReason, usage } = firstCandidate(answer);
  const { text, calls } = answerTextAndCalls(parts);
  const message: Json = { role: 'assistant', content: text };
  if (calls.length > 0) {
    message.tool_calls = calls.map((call, index) => ({
      id: `call_${call.name}_${index}`,
      type: 'function',
      function: { name: call.name, arguments: JSON.stringify(call.args) },
    }));
  }
  return {
    id: `chatcmpl-${typeof answer.modelVersion === 'string' ? answer.modelVersion : model}`,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: calls.length > 0 ? 'tool_calls' : geminiStopToOpenAI(finishReason), logprobs: null }],
    usage: {
      prompt_tokens: numberOrZero(usage.promptTokenCount),
      completion_tokens: numberOrZero(usage.candidatesTokenCount),
      total_tokens: numberOrZero(usage.totalTokenCount),
    },
  };
}

/** A Gemini JSON answer as an Anthropic Messages object naming `model`. */
export function geminiToAnthropic(answer: Json, model: string): Json {
  const { parts, finishReason, usage } = firstCandidate(answer);
  const { text, calls } = answerTextAndCalls(parts);
  const content: Json[] = [];
  if (text !== '') content.push({ type: 'text', text });
  for (const call of calls) {
    content.push({ type: 'tool_use', id: `toolu_${call.name}`, name: call.name, input: call.args });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: `msg_${typeof answer.modelVersion === 'string' ? answer.modelVersion : model}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: calls.length > 0 ? 'tool_use' : geminiStopToAnthropic(finishReason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.promptTokenCount), output_tokens: numberOrZero(usage.candidatesTokenCount) },
  };
}

/** One server-sent event line set: `data: <json>` (and an optional `event:` line). */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split buffered SSE text into its frames: each frame's event name and data payloads. */
export function parseSse(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/**
 * A buffered Gemini :streamGenerateContent body as OpenAI chat chunks naming `model`:
 * a role chunk, one content chunk a text piece, a finish chunk, a usage chunk, then
 * [DONE]. The usage chunk carries the final frame's usageMetadata, which is the
 * only usage the client checks on a stream.
 */
export function openAIEventsFromGeminiText(text: string, model: string): string[] {
  const events: string[] = [];
  let sentRole = false;
  let finishReason: unknown = 'STOP';
  let usage: Json = {};
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: `chatcmpl-${model}`,
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  for (const frame of parseSse(text)) {
    let payload: unknown;
    try {
      payload = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(payload)) continue;
    const { parts, finishReason: reason, usage: frameUsage } = firstCandidate(payload);
    if (isRecord(frameUsage) && Object.keys(frameUsage).length > 0) usage = frameUsage;
    if (reason !== undefined) finishReason = reason;
    const piece = geminiText(parts);
    if (piece !== '') {
      if (!sentRole) {
        sentRole = true;
        events.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
      }
      events.push(sseEvent(null, chunk({ content: piece }, null)));
    }
  }
  if (!sentRole) events.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  events.push(sseEvent(null, chunk({}, geminiStopToOpenAI(finishReason))));
  events.push(
    sseEvent(null, {
      id: `chatcmpl-${model}`,
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: numberOrZero(usage.promptTokenCount),
        completion_tokens: numberOrZero(usage.candidatesTokenCount),
        total_tokens: numberOrZero(usage.totalTokenCount),
      },
    }),
  );
  events.push('data: [DONE]\n\n');
  return events;
}

/**
 * A buffered Gemini :streamGenerateContent body as the Anthropic event sequence
 * naming `model`: message_start, one text content block with a delta a piece,
 * content_block_stop, message_delta with the stop reason and the output tokens,
 * then message_stop. The input tokens are only known from the final frame, so the
 * start carries 0 and the delta carries the output count the client checks.
 */
export function anthropicEventsFromGeminiText(text: string, model: string): string[] {
  const events: string[] = [];
  let started = false;
  let blockOpen = false;
  let finishReason: unknown = 'STOP';
  let outputTokens = 0;
  const start = (): void => {
    if (started) return;
    started = true;
    events.push(
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      }),
    );
  };
  for (const frame of parseSse(text)) {
    let payload: unknown;
    try {
      payload = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(payload)) continue;
    const { parts, finishReason: reason, usage: frameUsage } = firstCandidate(payload);
    if (reason !== undefined) finishReason = reason;
    if (isRecord(frameUsage)) outputTokens = numberOrZero(frameUsage.candidatesTokenCount);
    const piece = geminiText(parts);
    if (piece === '') continue;
    start();
    if (!blockOpen) {
      blockOpen = true;
      events.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
    }
    events.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  start();
  if (blockOpen) events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  events.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: geminiStopToAnthropic(finishReason), stop_sequence: null },
      usage: { output_tokens: outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return events;
}
