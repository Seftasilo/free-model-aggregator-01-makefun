// The Anthropic upstream wire, for an OpenAI-shaped client.
//
// When the router picks an Anthropic-wire provider for a client that speaks the
// OpenAI wire, the request is rewritten as a Messages request and the answer is
// rewritten back as a Chat Completions object, in JSON or in server-sent events.
// The translation keeps every field the provider echoes: the system prompt, the
// user text (images included), the tool names, the output cap and the stop
// sequences. A JSON-mode request never takes this path (the Anthropic wire cannot
// carry response_format, so its echo would say json_mode was absent); main.ts
// keeps such requests on the OpenAI and Gemini wires instead.
//
// Nothing here reads a clock or rolls dice: the same request and the same
// provider answer always give the same bytes back.

import { isRecord, type Json } from './providers.ts';
import { parseSse, sseEvent } from './gemini.ts';

/** A fixed epoch second for every OpenAI-shaped answer's `created`: nothing reads the clock. */
const FIXED_CREATED = 1767225600;

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The text of a string, or of a block array's text blocks. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One OpenAI content part as an Anthropic content block: text or an inline image. */
function anthropicBlockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const dataUrl = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
    if (dataUrl !== null) {
      return { type: 'image', source: { type: 'base64', media_type: dataUrl[1] ?? 'image/png', data: dataUrl[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** An Anthropic tool_choice for an OpenAI tool_choice value. */
function anthropicToolChoice(choice: unknown): Json | null {
  if (typeof choice === 'string') {
    if (choice === 'auto') return { type: 'auto' };
    if (choice === 'none') return { type: 'none' };
    if (choice === 'required') return { type: 'any' };
    return null;
  }
  if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    return { type: 'tool', name: choice.function.name };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming
 * `model`: system messages become the system prompt, user parts become content
 * blocks, assistant tool calls become tool_use blocks, tool messages become
 * tool_result blocks, and the output cap, sampling, stops and streaming pass
 * through.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemTexts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemTexts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: Json = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) input = parsed;
          } catch {
            input = {};
          }
        }
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input,
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) }],
      });
      continue;
    }
    const blocks: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') blocks.push({ type: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        const converted = anthropicBlockOf(block);
        if (converted !== null) blocks.push(converted);
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (systemTexts.length > 0) out.system = systemTexts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = Array.isArray(body.stop) ? body.stop : [];
  const sequences = (stops as unknown[]).filter((s): s is string => typeof s === 'string');
  if (sequences.length > 0) out.stop_sequences = sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const toolChoice = anthropicToolChoice(body.tool_choice);
  if (toolChoice !== null) out.tool_choice = toolChoice;
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic stop_reason as an OpenAI finish_reason. */
export function anthropicStopToOpenAI(reason: unknown): string {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** An Anthropic Messages object as an OpenAI Chat Completions object naming `model`. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
  const message: Json = { role: 'assistant', content: text };
  const calls = content.filter((b): b is Json => isRecord(b) && b.type === 'tool_use');
  if (calls.length > 0) {
    message.tool_calls = calls.map((call) => ({
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      type: 'function',
      function: { name: typeof call.name === 'string' ? call.name : '', arguments: JSON.stringify(isRecord(call.input) ? call.input : {}) },
    }));
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `chatcmpl-${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: calls.length > 0 ? 'tool_calls' : anthropicStopToOpenAI(body.stop_reason), logprobs: null }],
    usage: {
      prompt_tokens: numberOrZero(usage.input_tokens),
      completion_tokens: numberOrZero(usage.output_tokens),
      total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
    },
  };
}

/**
 * A buffered Anthropic SSE body as OpenAI chat chunks naming `model`: a role
 * chunk, one content chunk a text delta, a finish chunk, a usage chunk, then
 * [DONE]. Tool uses the provider streams become tool_calls chunks with their
 * arguments joined from the input_json_delta events.
 */
export function openAIEventsFromAnthropicText(text: string, model: string): string[] {
  const events: string[] = [];
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: `chatcmpl-${model}`,
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  let sentRole = false;
  let textOpen = false;
  let toolIndex = -1;
  let toolId = '';
  let toolName = '';
  let toolArgs = '';
  let stopReason: unknown = 'end_turn';
  let inputTokens = 0;
  let outputTokens = 0;
  const flushTool = (): void => {
    if (toolIndex === -1) return;
    events.push(
      sseEvent(
        null,
        chunk({ tool_calls: [{ index: toolIndex, id: toolId, type: 'function', function: { name: toolName, arguments: toolArgs } }] }, null),
      ),
    );
    toolIndex = -1;
    toolId = '';
    toolName = '';
    toolArgs = '';
  };
  for (const frame of parseSse(text)) {
    let payload: unknown;
    try {
      payload = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(payload)) continue;
    const event = frame.event ?? (typeof payload.type === 'string' ? payload.type : '');
    if (event === 'message_start' && isRecord(payload.message) && isRecord(payload.message.usage)) {
      inputTokens = numberOrZero(payload.message.usage.input_tokens);
      continue;
    }
    if (event === 'content_block_start' && isRecord(payload.content_block)) {
      if (payload.content_block.type === 'text') textOpen = true;
      if (payload.content_block.type === 'tool_use') {
        if (!sentRole) {
          sentRole = true;
          events.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
        }
        toolIndex = typeof payload.index === 'number' ? payload.index : 0;
        toolId = typeof payload.content_block.id === 'string' ? payload.content_block.id : 'toolu_unknown';
        toolName = typeof payload.content_block.name === 'string' ? payload.content_block.name : '';
        toolArgs = '';
      }
      continue;
    }
    if (event === 'content_block_delta' && isRecord(payload.delta)) {
      if (payload.delta.type === 'text_delta' && typeof payload.delta.text === 'string' && payload.delta.text !== '') {
        if (!sentRole) {
          sentRole = true;
          events.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
        }
        events.push(sseEvent(null, chunk({ content: payload.delta.text }, null)));
      }
      if (payload.delta.type === 'input_json_delta' && typeof payload.delta.partial_json === 'string') {
        toolArgs += payload.delta.partial_json;
      }
      continue;
    }
    if (event === 'content_block_stop') {
      if (textOpen) textOpen = false;
      flushTool();
      continue;
    }
    if (event === 'message_delta') {
      if (isRecord(payload.delta) && typeof payload.delta.stop_reason === 'string') stopReason = payload.delta.stop_reason;
      if (isRecord(payload.usage)) outputTokens = numberOrZero(payload.usage.output_tokens);
      continue;
    }
  }
  flushTool();
  if (!sentRole) events.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  events.push(sseEvent(null, chunk({}, anthropicStopToOpenAI(stopReason))));
  events.push(
    sseEvent(null, {
      id: `chatcmpl-${model}`,
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
    }),
  );
  events.push('data: [DONE]\n\n');
  return events;
}
