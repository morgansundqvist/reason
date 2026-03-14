/*
Package reason provides a hexagonal architecture framework for building LLM interactions with OpenAI's API.
It features automatic tool-calling loops, structured JSON output, and multi-turn conversations.

Basic Usage:

	client := reason.NewClient(apiKey)
	resp, err := client.SimpleQuery(ctx, "What is the capital of France?")
	if err != nil {
		log.Fatal(err)
	}
	println(resp.Content)

Tool Calling Loops:

	executor := func(toolName string, args map[string]interface{}) (string, error) {
		// Implement tool logic here
		return "result", nil
	}
	resp, err := client.QueryWithToolLoop(ctx, question, tools, executor)
*/
package reason

import (
	"context"

	"github.com/morgansundqvist/reason/internal/adapters"
	"github.com/morgansundqvist/reason/internal/application"
	"github.com/morgansundqvist/reason/internal/domain"
	"github.com/morgansundqvist/reason/internal/ports"
)

// Re-export domain types for public use
type (
	Message       = domain.Message
	Tool          = domain.Tool
	Parameter     = domain.Parameter
	ParameterType = domain.ParameterType
	Response      = domain.Response
	ToolCall      = domain.ToolCall
	Usage         = domain.Usage
	ToolExecutor  = domain.ToolExecutor
	Option        = domain.Option
	Effort        = domain.Effort
	CallConfig    = domain.CallConfig
)

// Re-export parameter type constants
const (
	TypeString  = domain.TypeString
	TypeNumber  = domain.TypeNumber
	TypeInteger = domain.TypeInteger
	TypeBoolean = domain.TypeBoolean
	TypeArray   = domain.TypeArray
	TypeObject  = domain.TypeObject
)

// Re-export effort constants
const (
	EffortLow    = domain.EffortLow
	EffortMedium = domain.EffortMedium
	EffortHigh   = domain.EffortHigh
)

// Re-export option builders
var (
	WithModel        = domain.WithModel
	WithTemperature  = domain.WithTemperature
	WithMaxTokens    = domain.WithMaxTokens
	WithSystemPrompt = domain.WithSystemPrompt
	WithStrictJSON   = domain.WithStrictJSON
	WithEffort       = domain.WithEffort
	WithRateLimitKey = domain.WithRateLimitKey
	WithThink        = domain.WithThink
	WithTimeout      = domain.WithTimeout
)

// Client wraps an LLM service and Reasoner for public use.
// It provides a unified interface for all LLM operations.
type Client struct {
	reasoner *application.Reasoner
	service  ports.LLMService
}

// NewClient creates a new reason client with the given OpenAI API key.
// By default, it uses the "gpt-4o" model; override with WithModel() option.
func NewClient(apiKey string, opts ...Option) *Client {
	cfg := &CallConfig{
		Model: "gpt-4o",
	}
	for _, opt := range opts {
		opt(cfg)
	}

	svc := adapters.NewOpenAIService(&adapters.OpenAIConfig{
		APIKey: apiKey,
		Model:  cfg.Model,
	})

	return &Client{
		reasoner: application.NewReasoner(svc),
		service:  svc,
	}
}

// NewGeminiClient creates a new reason client backed by Google's Gemini API.
// By default, it uses the "gemini-2.0-flash" model; override with WithModel() option.
func NewGeminiClient(ctx context.Context, apiKey string, opts ...Option) (*Client, error) {
	cfg := &CallConfig{
		Model: "gemini-3.1-flash-lite-preview",
	}
	for _, opt := range opts {
		opt(cfg)
	}

	svc, err := adapters.NewGeminiService(ctx, &adapters.GeminiConfig{
		APIKey: apiKey,
		Model:  cfg.Model,
	})
	if err != nil {
		return nil, err
	}

	return &Client{
		reasoner: application.NewReasoner(svc),
		service:  svc,
	}, nil
}

// NewOllamaClient creates a new reason client backed by Ollama's chat API.
// By default, it uses the "qwen3.5:4b" model; override with WithModel() option.
func NewOllamaClient(baseURL string, opts ...Option) (*Client, error) {
	cfg := &CallConfig{
		Model: "qwen3.5:4b",
	}
	for _, opt := range opts {
		opt(cfg)
	}

	svc, err := adapters.NewOllamaService(&adapters.OllamaConfig{
		BaseURL: baseURL,
		Model:   cfg.Model,
		Timeout: cfg.Timeout,
	})
	if err != nil {
		return nil, err
	}

	return &Client{
		reasoner: application.NewReasoner(svc),
		service:  svc,
	}, nil
}

// SimpleQuery asks a straightforward question without tools.
// Returns the LLM's text response.
func (c *Client) SimpleQuery(ctx context.Context, question string, opts ...Option) (*Response, error) {
	return c.reasoner.SimpleQuery(ctx, question, opts...)
}

// QueryWithTools asks a question and allows the LLM to invoke tools.
// The response will include any tool calls the LLM decided to make.
func (c *Client) QueryWithTools(ctx context.Context, question string, tools []Tool, opts ...Option) (*Response, error) {
	return c.reasoner.QueryWithTools(ctx, question, tools, opts...)
}

// StructuredQuery asks for a typed response conforming to a JSON schema.
// Useful for ensuring output matches a specific structure.
func (c *Client) StructuredQuery(ctx context.Context, question string, schema map[string]interface{}, opts ...Option) (*Response, error) {
	return c.reasoner.StructuredQuery(ctx, question, schema, opts...)
}

// StructuredQueryWithTools combines structured output with tool calling.
// The response will be a valid JSON string matching the schema and may include tool calls.
func (c *Client) StructuredQueryWithTools(ctx context.Context, question string, tools []Tool, schema map[string]interface{}, opts ...Option) (*Response, error) {
	return c.reasoner.StructuredQueryWithTools(ctx, question, tools, schema, opts...)
}

// Chat maintains a conversation across multiple turns.
// Pass in all messages from the conversation history; returns the assistant's next message.
func (c *Client) Chat(ctx context.Context, messages []Message, opts ...Option) (*Response, error) {
	return c.service.Chat(ctx, messages, opts...)
}

// ChatWithTools is Chat with tool support.
// The LLM can invoke tools within the conversation.
func (c *Client) ChatWithTools(ctx context.Context, messages []Message, tools []Tool, opts ...Option) (*Response, error) {
	return c.service.ChatWithTools(ctx, messages, tools, opts...)
}

// QueryWithToolLoop asks a question and automatically executes tools in a loop.
// The executor function is called for each tool the LLM wants to invoke.
// The loop continues until the LLM stops requesting tools or max iterations (5) is reached.
// Returns the final response after all tool execution is complete.
func (c *Client) QueryWithToolLoop(ctx context.Context, question string, tools []Tool, executor ToolExecutor, opts ...Option) (*Response, error) {
	return c.reasoner.QueryWithToolLoop(ctx, question, tools, executor, opts...)
}

// ChatWithToolLoop continues a conversation with automatic tool execution.
// Similar to QueryWithToolLoop but starts from an existing message history.
// The executor function is called for each tool invocation.
// Returns the final response after all tool execution is complete.
func (c *Client) ChatWithToolLoop(ctx context.Context, messages []Message, tools []Tool, executor ToolExecutor, opts ...Option) (*Response, error) {
	return c.reasoner.ChatWithToolLoop(ctx, messages, tools, executor, opts...)
}
