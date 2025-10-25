package application

import (
	"context"

	"reason/internal/domain"
	"reason/internal/ports"
)

// Reasoner encapsulates LLM reasoning workflows
type Reasoner struct {
	llm ports.LLMService
}

// NewReasoner creates a new Reasoner with the given LLM service
func NewReasoner(llm ports.LLMService) *Reasoner {
	return &Reasoner{llm: llm}
}

// SimpleQuery asks a straightforward question
func (r *Reasoner) SimpleQuery(ctx context.Context, question string, opts ...domain.Option) (*domain.Response, error) {
	return r.llm.AskQuestion(ctx, question, opts...)
}

// QueryWithTools asks a question and allows tool invocations
func (r *Reasoner) QueryWithTools(ctx context.Context, question string, tools []domain.Tool, opts ...domain.Option) (*domain.Response, error) {
	return r.llm.AskQuestionWithTools(ctx, question, tools, opts...)
}

// StructuredQuery asks for a typed response
func (r *Reasoner) StructuredQuery(ctx context.Context, question string, schema map[string]interface{}, opts ...domain.Option) (*domain.Response, error) {
	return r.llm.AskTypedQuestion(ctx, question, schema, opts...)
}

// StructuredQueryWithTools combines structured output with tool calling
func (r *Reasoner) StructuredQueryWithTools(ctx context.Context, question string, tools []domain.Tool, schema map[string]interface{}, opts ...domain.Option) (*domain.Response, error) {
	return r.llm.AskTypedQuestionWithTools(ctx, question, tools, schema, opts...)
}

// QueryWithToolLoop asks a question with automatic tool execution until no more tools are needed
func (r *Reasoner) QueryWithToolLoop(ctx context.Context, question string, tools []domain.Tool, executor domain.ToolExecutor, opts ...domain.Option) (*domain.Response, error) {
	return r.llm.RunToolLoop(ctx, question, tools, executor, opts...)
}

// ChatWithToolLoop continues a conversation with automatic tool execution until no more tools are needed
func (r *Reasoner) ChatWithToolLoop(ctx context.Context, messages []domain.Message, tools []domain.Tool, executor domain.ToolExecutor, opts ...domain.Option) (*domain.Response, error) {
	return r.llm.RunChatToolLoop(ctx, messages, tools, executor, opts...)
}
