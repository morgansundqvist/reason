# reason

A hexagonal architecture Go framework for building LLM interactions with OpenAI's API and Ollama's chat API, featuring automatic tool-calling loops and structured JSON output.

## Features

- **Automatic Tool-Calling Loops** — Multi-step tool execution with provider-specific message sequencing (max 5 iterations)
- **Structured Output** — JSON schema validation and typed responses
- **Multi-Turn Conversations** — Full message history management across multiple turns
- **Hexagonal Architecture** — Clean separation of concerns (domain → ports → adapters → application)
- **Framework-Agnostic Domain** — Easy to extend or swap LLM providers

## Quick Start

### Prerequisites

- Go 1.25.1 or later
- OpenAI API key for the OpenAI adapter
- A running Ollama server for the Ollama adapter

### Installation

```bash
go get github.com/morgansundqvist/reason
```

### Basic Usage

```go
package main

import (
	"context"
	"log"
	"os"

	"github.com/morgansundqvist/reason"
)

func main() {
	// Create a client
	client := reason.NewClient(os.Getenv("OPENAI_API_KEY"))
	ctx := context.Background()

	// Simple question
	resp, err := client.SimpleQuery(ctx, "What is the capital of France?")
	if err != nil {
		log.Fatal(err)
	}
	println(resp.Content) // Output: The capital of France is Paris.
}
```

### Ollama Usage

```go
package main

import (
	"context"
	"log"

	"github.com/morgansundqvist/reason"
)

func main() {
	client, err := reason.NewOllamaClient(
		"http://localhost:11434",
		reason.WithModel("qwen3.5:4b"),
	)
	if err != nil {
		log.Fatal(err)
	}

	resp, err := client.SimpleQuery(context.Background(), "What is the capital of France?")
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
adapters/        → Provider implementations (OpenAI SDK and Ollama HTTP)
application/     → Use case orchestration (Reasoner delegates to LLM service)
```

## API Methods

### Single-Turn Operations

- `SimpleQuery(ctx, question)` — Direct question
- `QueryWithTools(ctx, question, tools)` — Question with tool availability
- `StructuredQuery(ctx, question, jsonSchema)` — Typed JSON response
- `StructuredQueryWithTools(ctx, question, tools, jsonSchema)` — Typed response with tools
## API Methods

All methods are available on the `reason.Client` type created via `reason.NewClient(apiKey)`.

### Single-Turn Operations

- `SimpleQuery(ctx, question)` — Direct question without tools
- `QueryWithTools(ctx, question, tools)` — Question with tool availability
- `StructuredQuery(ctx, question, jsonSchema)` — Typed JSON response
- `StructuredQueryWithTools(ctx, question, tools, jsonSchema)` — Typed response with tools

## Configuration Options

```go
client := reason.NewClient(apiKey,
	reason.WithModel("gpt-4"),           // Override default model
	reason.WithTemperature(0.7),         // Randomness (0.0-1.0)
	reason.WithMaxTokens(2000),          // Output limit
	reason.WithSystemPrompt("Be helpful"), // Custom system prompt
	reason.WithStrictJSON(true),         // Enforce JSON format
	reason.WithEffort(reason.EffortLow), // Maps to reasoning effort / Ollama think
)

For Ollama, `WithEffort(reason.EffortLow|Medium|High)` is sent as the top-level `/api/chat` `think` field when the model supports string levels. The adapter is currently hardcoded to send `think: false` by default for testing.
## Tool Definition Example

```go
tools := []reason.Tool{
	{
		Name:        "calculator",
		Description: "Performs basic arithmetic",
		Parameters: []reason.Parameter{
			{
				Name:        "operation",
				Type:        reason.TypeString,
				Description: "add, subtract, multiply, divide",
				Required:    true,
				Enum:        []string{"add", "subtract", "multiply", "divide"},
			},
			{
				Name:        "a",
				Type:        reason.TypeNumber,
				Description: "First number",
				Required:    true,
			},
			{
				Name:        "b",
				Type:        reason.TypeNumber,
				Description: "Second number",
				Required:    true,
			},
		},
	},
}

// Single tool call
## Tool Executor Pattern

```go
// Define a tool executor - called for each tool the LLM invokes
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

// Use in tool loop - automatically calls executor for each tool invocation
resp, err := client.QueryWithToolLoop(ctx, 
	"Add 15+23, then multiply by 2",
	tools,
	executor,
	reason.WithTemperature(0.3),
)
```
	
	return fmt.Sprintf("Result: %v", result), nil
}

// Use in tool loop
resp, err := reasoner.QueryWithToolLoop(ctx, question, tools, executor)
```

## Running Tests

```bash
export OPENAI_API_KEY="sk-..."
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="qwen3.5:4b" # optional if using the adapter default
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
- Go standard library `net/http` — Ollama chat API client

## Key Design Decisions

### Tool Loops at Adapter Layer
Tool-calling loop logic lives in the adapter (not application layer) to properly manage each provider's message format requirements. The OpenAI adapter preserves `tool_calls` via `ChatCompletionMessage.ToParam()`, while the Ollama adapter maintains the `/api/chat` message history directly.

### Framework-Agnostic Domain
The `domain/` package contains no provider SDK types, making it easy to add other LLM providers in the future by implementing the `LLMService` interface.

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
