package ports

import (
	"context"
	"reason/internal/domain"
)

// LLMService defines the contract for LLM interactions
type LLMService interface {
	// AskQuestion handles simple text-only questions
	AskQuestion(ctx context.Context, question string, opts ...domain.Option) (*domain.Response, error)

	// AskQuestionWithTools allows the LLM to call tools
	AskQuestionWithTools(ctx context.Context, question string, tools []domain.Tool, opts ...domain.Option) (*domain.Response, error)

	// AskTypedQuestion asks a question expecting structured output
	AskTypedQuestion(ctx context.Context, question string, responseSchema map[string]interface{}, opts ...domain.Option) (*domain.Response, error)

	// AskTypedQuestionWithTools combines typed responses with tool calls
	AskTypedQuestionWithTools(ctx context.Context, question string, tools []domain.Tool, responseSchema map[string]interface{}, opts ...domain.Option) (*domain.Response, error)

	// Chat maintains a conversation across multiple turns
	Chat(ctx context.Context, messages []domain.Message, opts ...domain.Option) (*domain.Response, error)

	// ChatWithTools is Chat with tool support
	ChatWithTools(ctx context.Context, messages []domain.Message, tools []domain.Tool, opts ...domain.Option) (*domain.Response, error)

	// RunToolLoop executes a tool-calling loop: initial question -> execute tools -> feed results -> repeat
	// Manages message history internally and returns final response (max 5 iterations, stops when no more tool calls)
	RunToolLoop(ctx context.Context, question string, tools []domain.Tool, executor domain.ToolExecutor, opts ...domain.Option) (*domain.Response, error)

	// RunChatToolLoop executes a tool-calling loop with an existing message history
	// Manages message history internally and returns final response (max 5 iterations, stops when no more tool calls)
	RunChatToolLoop(ctx context.Context, messages []domain.Message, tools []domain.Tool, executor domain.ToolExecutor, opts ...domain.Option) (*domain.Response, error)
}
