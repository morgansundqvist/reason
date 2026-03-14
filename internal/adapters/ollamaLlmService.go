package adapters

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/morgansundqvist/reason/internal/domain"
	"github.com/morgansundqvist/reason/internal/ports"
)

const ollamaMaxToolIterations = 5

// OllamaService implements the ports.LLMService interface using Ollama's chat API.
type OllamaService struct {
	client *http.Client
	config *OllamaConfig
}

// OllamaConfig holds configuration for the Ollama service.
type OllamaConfig struct {
	BaseURL    string
	Model      string
	HTTPClient *http.Client
}

type ollamaChatRequest struct {
	Model    string                 `json:"model"`
	Messages []ollamaMessage        `json:"messages"`
	Tools    []ollamaTool           `json:"tools,omitempty"`
	Format   interface{}            `json:"format,omitempty"`
	Options  map[string]interface{} `json:"options,omitempty"`
	Think    interface{}            `json:"think,omitempty"`
	Stream   bool                   `json:"stream"`
}

type ollamaChatResponse struct {
	Model              string        `json:"model"`
	CreatedAt          string        `json:"created_at"`
	Message            ollamaMessage `json:"message"`
	Done               bool          `json:"done"`
	DoneReason         string        `json:"done_reason"`
	TotalDuration      int64         `json:"total_duration"`
	LoadDuration       int64         `json:"load_duration"`
	PromptEvalCount    int           `json:"prompt_eval_count"`
	PromptEvalDuration int64         `json:"prompt_eval_duration"`
	EvalCount          int           `json:"eval_count"`
	EvalDuration       int64         `json:"eval_duration"`
	Error              string        `json:"error,omitempty"`
}

type ollamaMessage struct {
	Role       string           `json:"role"`
	Content    string           `json:"content,omitempty"`
	Thinking   string           `json:"thinking,omitempty"`
	ToolCalls  []ollamaToolCall `json:"tool_calls,omitempty"`
	Name       string           `json:"name,omitempty"`
	ToolName   string           `json:"tool_name,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

type ollamaTool struct {
	Type     string               `json:"type"`
	Function ollamaToolDefinition `json:"function"`
}

type ollamaToolDefinition struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

type ollamaToolCall struct {
	ID       string               `json:"id,omitempty"`
	Type     string               `json:"type,omitempty"`
	Function ollamaToolInvocation `json:"function"`
}

type ollamaToolInvocation struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Arguments   map[string]interface{} `json:"arguments,omitempty"`
}

// NewOllamaService creates a new Ollama LLM service.
func NewOllamaService(cfg *OllamaConfig) (*OllamaService, error) {
	if cfg == nil {
		return nil, fmt.Errorf("ollama config is required")
	}
	if strings.TrimSpace(cfg.BaseURL) == "" {
		return nil, fmt.Errorf("ollama base URL is required")
	}

	parsedURL, err := url.Parse(cfg.BaseURL)
	if err != nil {
		return nil, fmt.Errorf("invalid ollama base URL: %w", err)
	}
	if parsedURL.Scheme == "" || parsedURL.Host == "" {
		return nil, fmt.Errorf("invalid ollama base URL: %q", cfg.BaseURL)
	}

	model := cfg.Model
	if strings.TrimSpace(model) == "" {
		model = "qwen3.5:4b"
	}

	client := cfg.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 2 * time.Minute}
	}

	return &OllamaService{
		client: client,
		config: &OllamaConfig{
			BaseURL:    strings.TrimRight(parsedURL.String(), "/"),
			Model:      model,
			HTTPClient: client,
		},
	}, nil
}

// AskQuestion handles simple text-only questions.
func (s *OllamaService) AskQuestion(ctx context.Context, question string, opts ...domain.Option) (*domain.Response, error) {
	cfg := s.applyOptions(opts)
	messages := s.buildMessages([]string{question}, cfg.SystemPrompt)
	return s.chat(ctx, messages, nil, nil, cfg)
}

// AskQuestionWithTools allows the LLM to call tools.
func (s *OllamaService) AskQuestionWithTools(ctx context.Context, question string, tools []domain.Tool, opts ...domain.Option) (*domain.Response, error) {
	cfg := s.applyOptions(opts)
	messages := s.buildMessages([]string{question}, cfg.SystemPrompt)
	return s.chat(ctx, messages, s.toolsToOllamaTools(tools), nil, cfg)
}

// AskTypedQuestion asks a question expecting structured output.
func (s *OllamaService) AskTypedQuestion(ctx context.Context, question string, responseSchema map[string]interface{}, opts ...domain.Option) (*domain.Response, error) {
	cfg := s.applyOptions(opts)
	messages := s.buildMessages([]string{question}, cfg.SystemPrompt)
	return s.chat(ctx, messages, nil, responseSchema, cfg)
}

// AskTypedQuestionWithTools combines typed responses with tool calls.
func (s *OllamaService) AskTypedQuestionWithTools(ctx context.Context, question string, tools []domain.Tool, responseSchema map[string]interface{}, opts ...domain.Option) (*domain.Response, error) {
	cfg := s.applyOptions(opts)
	messages := s.buildMessages([]string{question}, cfg.SystemPrompt)
	return s.chat(ctx, messages, s.toolsToOllamaTools(tools), responseSchema, cfg)
}

// Chat maintains a conversation across multiple turns.
func (s *OllamaService) Chat(ctx context.Context, messages []domain.Message, opts ...domain.Option) (*domain.Response, error) {
	cfg := s.applyOptions(opts)
	chatMessages := s.messagesToOllamaMessages(messages)
	chatMessages = s.prependSystemPrompt(chatMessages, cfg.SystemPrompt)
	return s.chat(ctx, chatMessages, nil, nil, cfg)
}

// ChatWithTools is Chat with tool support.
func (s *OllamaService) ChatWithTools(ctx context.Context, messages []domain.Message, tools []domain.Tool, opts ...domain.Option) (*domain.Response, error) {
	cfg := s.applyOptions(opts)
	chatMessages := s.messagesToOllamaMessages(messages)
	chatMessages = s.prependSystemPrompt(chatMessages, cfg.SystemPrompt)
	return s.chat(ctx, chatMessages, s.toolsToOllamaTools(tools), nil, cfg)
}

// RunToolLoop executes a tool-calling loop for an initial user question.
func (s *OllamaService) RunToolLoop(ctx context.Context, question string, tools []domain.Tool, executor domain.ToolExecutor, opts ...domain.Option) (*domain.Response, error) {
	cfg := s.applyOptions(opts)
	messages := s.buildMessages([]string{question}, cfg.SystemPrompt)
	return s.runToolLoop(ctx, messages, s.toolsToOllamaTools(tools), executor, cfg)
}

// RunChatToolLoop executes a tool-calling loop with an existing message history.
func (s *OllamaService) RunChatToolLoop(ctx context.Context, messages []domain.Message, tools []domain.Tool, executor domain.ToolExecutor, opts ...domain.Option) (*domain.Response, error) {
	cfg := s.applyOptions(opts)
	chatMessages := s.messagesToOllamaMessages(messages)
	chatMessages = s.prependSystemPrompt(chatMessages, cfg.SystemPrompt)
	return s.runToolLoop(ctx, chatMessages, s.toolsToOllamaTools(tools), executor, cfg)
}

func (s *OllamaService) chat(ctx context.Context, messages []ollamaMessage, tools []ollamaTool, responseSchema map[string]interface{}, cfg *domain.CallConfig) (*domain.Response, error) {
	chatResp, err := s.sendChatRequest(ctx, messages, tools, responseSchema, cfg)
	if err != nil {
		return nil, err
	}

	return s.chatResponseToDomain(chatResp), nil
}

func (s *OllamaService) runToolLoop(ctx context.Context, messages []ollamaMessage, tools []ollamaTool, executor domain.ToolExecutor, cfg *domain.CallConfig) (*domain.Response, error) {
	chatResp, err := s.sendChatRequest(ctx, messages, tools, nil, cfg)
	if err != nil {
		return nil, err
	}

	resp := s.chatResponseToDomain(chatResp)

	for iteration := 0; iteration < ollamaMaxToolIterations; iteration++ {
		if len(chatResp.Message.ToolCalls) == 0 {
			break
		}

		messages = append(messages, chatResp.Message)

		for _, toolCall := range chatResp.Message.ToolCalls {
			result, execErr := executor(toolCall.Function.Name, toolCall.Function.Arguments)
			if execErr != nil {
				result = "Error: " + execErr.Error()
			}

			messages = append(messages, ollamaMessage{
				Role:       "tool",
				Content:    result,
				Name:       toolCall.Function.Name,
				ToolName:   toolCall.Function.Name,
				ToolCallID: toolCall.ID,
			})
		}

		chatResp, err = s.sendChatRequest(ctx, messages, tools, nil, cfg)
		if err != nil {
			return nil, err
		}

		resp = s.chatResponseToDomain(chatResp)
	}

	return resp, nil
}

func (s *OllamaService) sendChatRequest(ctx context.Context, messages []ollamaMessage, tools []ollamaTool, responseSchema map[string]interface{}, cfg *domain.CallConfig) (*ollamaChatResponse, error) {
	requestBody := s.buildChatRequest(messages, tools, responseSchema, cfg)
	body, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("marshal ollama request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.config.BaseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create ollama request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	res, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama request failed: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode < http.StatusOK || res.StatusCode >= http.StatusMultipleChoices {
		body, _ := io.ReadAll(io.LimitReader(res.Body, 1<<20))
		message := strings.TrimSpace(string(body))
		if message == "" {
			message = http.StatusText(res.StatusCode)
		}
		return nil, fmt.Errorf("ollama chat request failed with status %s: %s", res.Status, message)
	}

	var chatResp ollamaChatResponse
	if err := json.NewDecoder(res.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("decode ollama response: %w", err)
	}
	if chatResp.Error != "" {
		return nil, fmt.Errorf("ollama error: %s", chatResp.Error)
	}

	s.normalizeToolCalls(&chatResp.Message)

	return &chatResp, nil
}

func (s *OllamaService) buildChatRequest(messages []ollamaMessage, tools []ollamaTool, responseSchema map[string]interface{}, cfg *domain.CallConfig) *ollamaChatRequest {
	req := &ollamaChatRequest{
		Model:    cfg.Model,
		Messages: messages,
		Think:    cfg.Think,
		Stream:   false,
	}

	if len(tools) > 0 {
		req.Tools = tools
	}

	if responseSchema != nil {
		req.Format = responseSchema
	} else if cfg.StrictJSON {
		req.Format = "json"
	}

	options := make(map[string]interface{})
	if cfg.Temperature > 0 {
		options["temperature"] = cfg.Temperature
	}
	if cfg.MaxTokens > 0 {
		options["num_predict"] = cfg.MaxTokens
	}
	if len(options) > 0 {
		req.Options = options
	}

	return req
}

func (s *OllamaService) buildMessages(questions []string, systemPrompt string) []ollamaMessage {
	messages := make([]ollamaMessage, 0, len(questions)+1)

	if systemPrompt != "" {
		messages = append(messages, ollamaMessage{Role: "system", Content: systemPrompt})
	}

	for _, question := range questions {
		messages = append(messages, ollamaMessage{Role: "user", Content: question})
	}

	return messages
}

func (s *OllamaService) prependSystemPrompt(messages []ollamaMessage, systemPrompt string) []ollamaMessage {
	if systemPrompt == "" {
		return messages
	}
	if len(messages) > 0 && messages[0].Role == "system" {
		return messages
	}

	return append([]ollamaMessage{{Role: "system", Content: systemPrompt}}, messages...)
}

func (s *OllamaService) applyOptions(opts []domain.Option) *domain.CallConfig {
	cfg := &domain.CallConfig{Model: s.config.Model}
	for _, opt := range opts {
		opt(cfg)
	}
	if cfg.Model == "" {
		cfg.Model = s.config.Model
	}
	return cfg
}

func (s *OllamaService) chatResponseToDomain(chatResp *ollamaChatResponse) *domain.Response {
	resp := &domain.Response{
		Content:    chatResp.Message.Content,
		StopReason: chatResp.DoneReason,
		Usage: &domain.Usage{
			InputTokens:  chatResp.PromptEvalCount,
			OutputTokens: chatResp.EvalCount,
		},
	}

	if len(chatResp.Message.ToolCalls) > 0 {
		resp.RawMessage = map[string]interface{}{
			"role":       chatResp.Message.Role,
			"content":    chatResp.Message.Content,
			"tool_calls": chatResp.Message.ToolCalls,
		}
	}

	for _, toolCall := range chatResp.Message.ToolCalls {
		resp.ToolCalls = append(resp.ToolCalls, domain.ToolCall{
			ID:        toolCall.ID,
			Name:      toolCall.Function.Name,
			Arguments: toolCall.Function.Arguments,
		})
	}

	return resp
}

func (s *OllamaService) normalizeToolCalls(message *ollamaMessage) {
	for index := range message.ToolCalls {
		if message.ToolCalls[index].ID == "" {
			message.ToolCalls[index].ID = uuid.NewString()
		}
		if message.ToolCalls[index].Type == "" {
			message.ToolCalls[index].Type = "function"
		}
		if message.ToolCalls[index].Function.Arguments == nil {
			message.ToolCalls[index].Function.Arguments = map[string]interface{}{}
		}
	}
}

func (s *OllamaService) toolsToOllamaTools(tools []domain.Tool) []ollamaTool {
	toolDefs := make([]ollamaTool, 0, len(tools))

	for _, tool := range tools {
		toolDefs = append(toolDefs, ollamaTool{
			Type: "function",
			Function: ollamaToolDefinition{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  s.parametersToJSONSchema(tool.Parameters),
			},
		})
	}

	return toolDefs
}

func (s *OllamaService) parametersToJSONSchema(params []domain.Parameter) map[string]interface{} {
	properties := make(map[string]interface{}, len(params))
	required := make([]string, 0, len(params))

	for _, param := range params {
		property := map[string]interface{}{
			"type":        string(param.Type),
			"description": param.Description,
		}

		if len(param.Enum) > 0 {
			property["enum"] = param.Enum
		}

		properties[param.Name] = property
		if param.Required {
			required = append(required, param.Name)
		}
	}

	return map[string]interface{}{
		"type":                 "object",
		"properties":           properties,
		"required":             required,
		"additionalProperties": false,
	}
}

func (s *OllamaService) messagesToOllamaMessages(messages []domain.Message) []ollamaMessage {
	ollamaMessages := make([]ollamaMessage, 0, len(messages))

	for _, message := range messages {
		ollamaMessage := ollamaMessage{
			Role:    message.Role,
			Content: message.Content,
		}

		if len(message.ToolCalls) > 0 {
			ollamaMessage.ToolCalls = s.domainToolCallsToOllamaToolCalls(message.ToolCalls)
		}

		ollamaMessages = append(ollamaMessages, ollamaMessage)
	}

	return ollamaMessages
}

func (s *OllamaService) domainToolCallsToOllamaToolCalls(toolCalls []domain.ToolCall) []ollamaToolCall {
	converted := make([]ollamaToolCall, 0, len(toolCalls))

	for _, toolCall := range toolCalls {
		converted = append(converted, ollamaToolCall{
			ID:   toolCall.ID,
			Type: "function",
			Function: ollamaToolInvocation{
				Name:      toolCall.Name,
				Arguments: toolCall.Arguments,
			},
		})
	}

	return converted
}

var _ ports.LLMService = (*OllamaService)(nil)
