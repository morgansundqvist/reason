# Copilot Instructions for "reason" - LLM Service Framework

## Project Overview

This is a **hexagonal architecture** Go project for building LLM interactions with OpenAI's API. The codebase implements automatic tool-calling loops, structured JSON output, and multi-turn conversations with clean separation of concerns across four layers.

## Architecture Layers

```
domain/          → Business logic & types (Tool, Message, Response, ToolCall, ToolExecutor)
ports/           → Interface contracts (LLMService interface defines all operations)
adapters/        → OpenAI SDK implementation (OpenAIService wraps github.com/openai/openai-go/v3)
application/     → Use case orchestration (Reasoner delegates to LLM service)
```

**Key principle**: Domain models are framework-agnostic. The adapter layer handles SDK-specific concerns and message format conversions.

## Critical Architectural Pattern: Tool-Calling Loops

This is the most complex feature and requires understanding OpenAI's message sequencing:

1. **Problem**: OpenAI's tool-calling loop requires careful message state management:
   - Assistant message with `tool_calls` must precede tool messages
   - Tool messages must include the exact `tool_call_id` from the assistant response
   - Loop continues while `tool_calls` exist; stops when empty or max iterations reached

2. **Solution**: Tool loop logic lives in the **adapter layer** (not application), because:
   - Raw OpenAI message format must be preserved across iterations (`ChatCompletionMessage` → `.ToParam()`)
   - Message history must remain in OpenAI's format to properly reconstruct API payloads
   - Application layer delegates to `RunToolLoop()` / `RunChatToolLoop()` port methods

3. **Implementation details** (`openaiLlmService.go` `RunToolLoop` method):
   - Store `lastChoice *openai.ChatCompletionMessage` from each API response
   - Use `lastChoice.ToParam()` to append the assistant message (preserves `tool_calls`)
   - For each tool call, execute it via `ToolExecutor` callback, then append via `openai.ToolMessage(result, toolCallID)` 
   - **Critical**: Parameter order to `ToolMessage()` is `(content, toolCallID)` — reversed order causes OpenAI API errors
   - Maximum 5 iterations to prevent infinite loops; return when `tool_calls` is empty

4. **Test 5** (`cmd/test/main.go::testToolCallingLoop`): Validates this works with multi-step arithmetic: "add 15+23 → multiply by 2" triggers two consecutive tool calls.

## Key Integration Points

### Domain Models (`internal/domain/llm.go`)

- **`Message`**: Conversation turn. Only assistant messages have `ToolCalls` field.
- **`Tool`**: Defines a callable function with name, description, and typed `Parameters` (using JSON schema types).
- **`Parameter`**: Input schema with `Type` (string/number/integer/boolean/array/object), `Description`, `Required`, optional `Enum`.
- **`ToolCall`**: Result of LLM deciding to call a tool. Contains `ID` (unique per call), `Name`, and `Arguments` (parsed JSON map).
- **`ToolExecutor`**: Callback function `func(toolName string, args map[string]interface{}) (string, error)` — implement this to execute tools.
- **`CallConfig`**: Configuration options (model, temperature, maxTokens, systemPrompt, strictJSON).

### Ports (`internal/ports/llmService.go`)

All LLM operations route through the `LLMService` interface:

- **Single-turn**: `AskQuestion()`, `AskQuestionWithTools()`, `AskTypedQuestion()`, `AskTypedQuestionWithTools()`
- **Multi-turn**: `Chat()`, `ChatWithTools()`
- **Tool loops** (auto-execute): `RunToolLoop(question, tools, executor)`, `RunChatToolLoop(messages, tools, executor)`

Tool loop methods handle the iteration internally—pass a `ToolExecutor` callback and get back the final response.

### Adapter (`internal/adapters/openaiLlmService.go`)

All methods follow this pattern:

1. Apply options to `CallConfig` via `s.applyOptions(opts)`
2. Convert domain models to OpenAI SDK types:
   - Domain `Message[]` → `openai.ChatCompletionMessageParam[]` via `s.messagesToOpenAIMessages()`
   - Domain `Tool[]` → OpenAI tool definitions via `s.toolsToOpenAITools()`
3. Build request params via `s.buildParams(messages, tools, schema, cfg)`
4. Call `s.client.Chat.Completions.New(ctx, params)`
5. Convert response via `s.completionToResponse(completion)` back to domain `Response`

Helper methods centralize conversions and reduce duplication.

### Application (`internal/application/reasoner.go`)

The `Reasoner` is a thin wrapper that delegates all calls to the injected `LLMService`. No logic here—just forwarding with convenient method names.

## Developer Workflows

### Setup

```bash
export OPENAI_API_KEY="sk-..."
cd /home/morgan/dev/ai/reason
```

### Running Tests

```bash
go run cmd/test/main.go
```

Tests 1–4 validate individual operations; Test 5 validates the tool-calling loop. Exit code 0 = success.

### Adding New LLM Operations

1. **Define domain contract**: Add method to `LLMService` interface in `ports/llmService.go`
2. **Implement in adapter**: Add method to `OpenAIService` in `adapters/openaiLlmService.go` following the pattern: apply options → convert types → call SDK → convert response
3. **Add application wrapper**: Add delegation method in `application/reasoner.go`
4. **Add integration test**: Add test function in `cmd/test/main.go` and call from `main()`

### Debugging Tool Loops

If `RunToolLoop` or `RunChatToolLoop` fails:

- Check error message from OpenAI API (appears in test output)
- Common issues:
  - `"tool_call_id" not found`: Parameter order to `openai.ToolMessage()` is wrong → should be `(result, toolCallID)`
  - `tool messages not allowed without preceding assistant message`: Not appending assistant message via `.ToParam()` before tool messages
  - Tool executor returning error: Check `ToolExecutor` callback implementation and error handling in loop

## Go Module Details

- **Module**: `reason`
- **Go version**: 1.25.1
- **Main dependency**: `github.com/openai/openai-go/v3` (v3.6.1)
- **Build**: `go build` or `go run cmd/test/main.go`

## File Organization

```
go.mod                              # Module definition
cmd/test/main.go                    # Integration tests (all 5 tests here)
internal/
  domain/llm.go                     # Core types: Message, Tool, ToolCall, Response, etc.
  ports/llmService.go               # LLMService interface contract
  adapters/openaiLlmService.go      # OpenAI SDK implementation (~450 LOC)
  application/reasoner.go           # Reasoner wrapper for use cases
  models/                           # Reserved for domain-specific models
```

## When Implementing Features

✅ **DO**:
- Keep domain models framework-agnostic (no SDK types in `domain/`)
- Implement logic at the right layer (tool loops in adapter, not application)
- Use `.ToParam()` when reconstructing messages for API continuity (especially in tool loops)
- Add integration tests alongside new operations

❌ **DON'T**:
- Mix OpenAI SDK types into domain models
- Repeat conversion logic—extract to helper methods in adapter
- Forget parameter order in `openai.ToolMessage(content, id)` — it's easy to flip
- Allow tool loops to run indefinitely—enforce max iterations (5 is current standard)
