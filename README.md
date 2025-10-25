# reason

A hexagonal architecture Go framework for building LLM interactions with OpenAI's API, featuring automatic tool-calling loops and structured JSON output.

## Features

- **Automatic Tool-Calling Loops** — Multi-step tool execution with proper OpenAI message sequencing (max 5 iterations)
- **Structured Output** — JSON schema validation and typed responses
- **Multi-Turn Conversations** — Full message history management across multiple turns
- **Hexagonal Architecture** — Clean separation of concerns (domain → ports → adapters → application)
- **Framework-Agnostic Domain** — Easy to extend or swap LLM providers

## Quick Start

### Prerequisites

- Go 1.25.1 or later
- OpenAI API key

### Installation

```bash
go get your-gitea-server/your-username/reason
```

### Basic Usage

```go
package main

import (
	"context"
	"log"
	"os"

	"reason/internal/adapters"
	"reason/internal/application"
	"reason/internal/domain"
)

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	
	// Initialize service
	svc := adapters.NewOpenAIService(&adapters.OpenAIConfig{
		APIKey: apiKey,
		Model:  "gpt-4",
	})
	reasoner := application.NewReasoner(svc)
	ctx := context.Background()

	// Simple question
	resp, err := reasoner.SimpleQuery(ctx, "What is the capital of France?")
	if err != nil {
		log.Fatal(err)
	}
	println(resp.Content)
}
```

## Architecture

```
domain/          → Business logic & types (Tool, Message, Response, ToolCall, ToolExecutor)
ports/           → Interface contracts (LLMService interface defines all operations)
adapters/        → OpenAI SDK implementation (OpenAIService wraps github.com/openai/openai-go/v3)
application/     → Use case orchestration (Reasoner delegates to LLM service)
```

## API Methods

### Single-Turn Operations

- `SimpleQuery(ctx, question)` — Direct question
- `QueryWithTools(ctx, question, tools)` — Question with tool availability
- `StructuredQuery(ctx, question, jsonSchema)` — Typed JSON response
- `StructuredQueryWithTools(ctx, question, tools, jsonSchema)` — Typed response with tools

### Multi-Turn Operations

- `Chat(ctx, messages)` — Conversation management
- `ChatWithTools(ctx, messages, tools)` — Conversation with tool support

### Tool-Calling Loops (Automatic)

- `QueryWithToolLoop(ctx, question, tools, executor)` — Auto-execute tools until complete
- `ChatWithToolLoop(ctx, messages, tools, executor)` — Continue conversation with auto tool execution

## Configuration Options

```go
domain.WithModel("gpt-4")              // Override model
domain.WithTemperature(0.7)            // Randomness (0.0-1.0)
domain.WithMaxTokens(2000)             // Output limit
domain.WithSystemPrompt("Be helpful")  // Custom system prompt
domain.WithStrictJSON(true)            // Enforce JSON format
```

## Tool Definition Example

```go
tools := []domain.Tool{
	{
		Name:        "calculator",
		Description: "Performs basic arithmetic",
		Parameters: []domain.Parameter{
			{
				Name:        "operation",
				Type:        domain.TypeString,
				Description: "add, subtract, multiply, divide",
				Required:    true,
				Enum:        []string{"add", "subtract", "multiply", "divide"},
			},
			{
				Name:        "a",
				Type:        domain.TypeNumber,
				Description: "First number",
				Required:    true,
			},
			{
				Name:        "b",
				Type:        domain.TypeNumber,
				Description: "Second number",
				Required:    true,
			},
		},
	},
}
```

## Tool Executor Pattern

```go
executor := func(toolName string, args map[string]interface{}) (string, error) {
	if toolName != "calculator" {
		return "", fmt.Errorf("unknown tool: %s", toolName)
	}
	
	operation := args["operation"].(string)
	a := args["a"].(float64)
	b := args["b"].(float64)
	
	var result float64
	switch operation {
	case "add":
		result = a + b
	case "multiply":
		result = a * b
	// ... handle other operations
	}
	
	return fmt.Sprintf("Result: %v", result), nil
}

// Use in tool loop
resp, err := reasoner.QueryWithToolLoop(ctx, question, tools, executor)
```

## Running Tests

```bash
export OPENAI_API_KEY="sk-..."
go run cmd/test/main.go
```

**Tests included:**
- Test 1: Simple question answering
- Test 2: Single tool invocation
- Test 3: Structured JSON output
- Test 4: Multi-turn conversation
- Test 5: Tool-calling loop (multi-step tool execution)

## Dependencies

- `github.com/openai/openai-go/v3` — Official OpenAI Go SDK

## Key Design Decisions

### Tool Loops at Adapter Layer
Tool-calling loop logic lives in the adapter (not application layer) to properly manage OpenAI's message format requirements. This preserves the `tool_calls` field across iterations using `ChatCompletionMessage.ToParam()`.

### Framework-Agnostic Domain
The `domain/` package contains no OpenAI SDK types, making it easy to add other LLM providers in the future by implementing the `LLMService` interface.

### Functional Options Pattern
Configuration uses functional options (`WithModel()`, `WithTemperature()`, etc.) for clean, extensible API design.

## Common Issues

**Tool-calling fails with "tool_call_id not found"**
- Ensure `ToolMessage()` parameters are in correct order: `(content, toolCallID)` not `(toolCallID, content)`

**Tool messages not allowed error**
- The assistant message with `tool_calls` must be appended before tool messages using `.ToParam()`

**Infinite tool loop**
- Max iterations is 5 by design to prevent runaway loops; check your tool executor for conditions that should exit

## Contributing

See `.github/copilot-instructions.md` for developer workflow details and architectural patterns.

## License

MIT License — see LICENSE file for details

## Module Path

When cloning to a local Gitea server, update the module path in `go.mod`:

```go
module gitea.your-server/your-username/reason
```

Then import via:
```go
import "gitea.your-server/your-username/reason/internal/domain"
```
