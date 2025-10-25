package adapters

import (
	"context"
	"encoding/json"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	"reason/internal/domain"
	"reason/internal/ports"
)

// OpenAIService implements the ports.LLMService interface using OpenAI's API
type OpenAIService struct {
	client *openai.Client
	config *OpenAIConfig
}

// OpenAIConfig holds configuration for the OpenAI service
type OpenAIConfig struct {
	APIKey string
	Model  string // default model to use
}

// NewOpenAIService creates a new OpenAI LLM service
func NewOpenAIService(cfg *OpenAIConfig) *OpenAIService {
	client := openai.NewClient(option.WithAPIKey(cfg.APIKey))
	return &OpenAIService{
		client: &client,
		config: cfg,
	}
}

// AskQuestion handles simple text-only questions
func (s *OpenAIService) AskQuestion(ctx context.Context, question string, opts ...domain.Option) (*domain.Response, error) {
	cfg := s.applyOptions(opts)
	messages := s.buildMessages([]string{question}, cfg.SystemPrompt)
	params := s.buildParams(messages, nil, nil, cfg)

	completion, err := s.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, err
	}

	return s.completionToResponse(completion), nil
}

// AskQuestionWithTools allows the LLM to call tools
func (s *OpenAIService) AskQuestionWithTools(ctx context.Context, question string, tools []domain.Tool, opts ...domain.Option) (*domain.Response, error) {
	cfg := s.applyOptions(opts)
	messages := s.buildMessages([]string{question}, cfg.SystemPrompt)
	toolParams := s.toolsToOpenAITools(tools)
	params := s.buildParams(messages, toolParams, nil, cfg)

	completion, err := s.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, err
	}

	return s.completionToResponse(completion), nil
}

// AskTypedQuestion asks a question expecting structured output
func (s *OpenAIService) AskTypedQuestion(ctx context.Context, question string, responseSchema map[string]interface{}, opts ...domain.Option) (*domain.Response, error) {
	cfg := s.applyOptions(opts)
	messages := s.buildMessages([]string{question}, cfg.SystemPrompt)
	params := s.buildParams(messages, nil, responseSchema, cfg)

	completion, err := s.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, err
	}

	return s.completionToResponse(completion), nil
}

// AskTypedQuestionWithTools combines typed responses with tool calls
func (s *OpenAIService) AskTypedQuestionWithTools(ctx context.Context, question string, tools []domain.Tool, responseSchema map[string]interface{}, opts ...domain.Option) (*domain.Response, error) {
	cfg := s.applyOptions(opts)
	messages := s.buildMessages([]string{question}, cfg.SystemPrompt)
	toolParams := s.toolsToOpenAITools(tools)
	params := s.buildParams(messages, toolParams, responseSchema, cfg)

	completion, err := s.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, err
	}

	return s.completionToResponse(completion), nil
}

// Chat maintains a conversation across multiple turns
func (s *OpenAIService) Chat(ctx context.Context, messages []domain.Message, opts ...domain.Option) (*domain.Response, error) {
	cfg := s.applyOptions(opts)
	chatMessages := s.messagesToOpenAIMessages(messages)
	chatMessages = s.prependSystemPrompt(chatMessages, cfg.SystemPrompt)
	params := s.buildParams(chatMessages, nil, nil, cfg)

	completion, err := s.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, err
	}

	return s.completionToResponse(completion), nil
}

// ChatWithTools is Chat with tool support
func (s *OpenAIService) ChatWithTools(ctx context.Context, messages []domain.Message, tools []domain.Tool, opts ...domain.Option) (*domain.Response, error) {
	cfg := s.applyOptions(opts)
	chatMessages := s.messagesToOpenAIMessages(messages)
	chatMessages = s.prependSystemPrompt(chatMessages, cfg.SystemPrompt)
	toolParams := s.toolsToOpenAITools(tools)
	params := s.buildParams(chatMessages, toolParams, nil, cfg)

	completion, err := s.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, err
	}

	return s.completionToResponse(completion), nil
}

// Helper methods

func (s *OpenAIService) buildParams(
	messages []openai.ChatCompletionMessageParamUnion,
	tools []openai.ChatCompletionToolUnionParam,
	responseSchema map[string]interface{},
	cfg *domain.CallConfig,
) openai.ChatCompletionNewParams {
	params := openai.ChatCompletionNewParams{
		Messages: messages,
		Model:    openai.ChatModelGPT4o,
	}

	if len(tools) > 0 {
		params.Tools = tools
	}

	if responseSchema != nil {
		params.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONSchema: &openai.ResponseFormatJSONSchemaParam{
				JSONSchema: openai.ResponseFormatJSONSchemaJSONSchemaParam{
					Name:        "response",
					Description: openai.String("Structured response"),
					Schema:      responseSchema,
					Strict:      openai.Bool(cfg.StrictJSON),
				},
			},
		}
	}

	if cfg.Temperature > 0 {
		params.Temperature = openai.Float(cfg.Temperature)
	}
	if cfg.MaxTokens > 0 {
		params.MaxTokens = openai.Int(int64(cfg.MaxTokens))
	}

	return params
}

func (s *OpenAIService) buildMessages(questions []string, systemPrompt string) []openai.ChatCompletionMessageParamUnion {
	var messages []openai.ChatCompletionMessageParamUnion

	if systemPrompt != "" {
		messages = append(messages, openai.SystemMessage(systemPrompt))
	}

	for _, q := range questions {
		messages = append(messages, openai.UserMessage(q))
	}

	return messages
}

func (s *OpenAIService) prependSystemPrompt(
	messages []openai.ChatCompletionMessageParamUnion,
	systemPrompt string,
) []openai.ChatCompletionMessageParamUnion {
	if systemPrompt == "" || len(messages) == 0 {
		return messages
	}

	// Check if first message is already a system message
	if len(messages) > 0 {
		return append([]openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(systemPrompt),
		}, messages...)
	}

	return messages
}

func (s *OpenAIService) applyOptions(opts []domain.Option) *domain.CallConfig {
	cfg := &domain.CallConfig{
		Model: s.config.Model,
	}
	for _, opt := range opts {
		opt(cfg)
	}
	if cfg.Model == "" {
		cfg.Model = s.config.Model
	}
	return cfg
}

func (s *OpenAIService) completionToResponse(completion *openai.ChatCompletion) *domain.Response {
	if len(completion.Choices) == 0 {
		return &domain.Response{}
	}

	choice := completion.Choices[0]
	resp := &domain.Response{
		Content:    choice.Message.Content,
		StopReason: string(choice.FinishReason),
		Usage: &domain.Usage{
			InputTokens:  int(completion.Usage.PromptTokens),
			OutputTokens: int(completion.Usage.CompletionTokens),
		},
	}

	// Capture raw message data for tool_calls info
	if len(choice.Message.ToolCalls) > 0 {
		rawMsg := map[string]interface{}{
			"role":       "assistant",
			"content":    choice.Message.Content,
			"tool_calls": choice.Message.ToolCalls,
		}
		resp.RawMessage = rawMsg
	}

	// Convert tool calls if present
	for _, tc := range choice.Message.ToolCalls {
		var args map[string]interface{}
		_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)

		resp.ToolCalls = append(resp.ToolCalls, domain.ToolCall{
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Arguments: args,
		})
	}

	return resp
}

func (s *OpenAIService) toolsToOpenAITools(tools []domain.Tool) []openai.ChatCompletionToolUnionParam {
	var toolParams []openai.ChatCompletionToolUnionParam

	for _, tool := range tools {
		// Build JSON schema from typed Parameters
		schema := s.parametersToJSONSchema(tool.Parameters)
		params := openai.FunctionParameters(schema)

		toolParams = append(toolParams, openai.ChatCompletionToolUnionParam{
			OfFunction: &openai.ChatCompletionFunctionToolParam{
				Function: openai.FunctionDefinitionParam{
					Name:        tool.Name,
					Description: openai.String(tool.Description),
					Parameters:  params,
				},
			},
		})
	}

	return toolParams
}

// parametersToJSONSchema converts a slice of Parameters to a JSON schema
func (s *OpenAIService) parametersToJSONSchema(params []domain.Parameter) map[string]interface{} {
	properties := make(map[string]interface{})
	required := make([]string, 0)

	for _, p := range params {
		prop := map[string]interface{}{
			"type":        string(p.Type),
			"description": p.Description,
		}

		if len(p.Enum) > 0 {
			prop["enum"] = p.Enum
		}

		properties[p.Name] = prop

		if p.Required {
			required = append(required, p.Name)
		}
	}

	return map[string]interface{}{
		"type":                 "object",
		"properties":           properties,
		"required":             required,
		"additionalProperties": false,
	}
}

func (s *OpenAIService) messagesToOpenAIMessages(messages []domain.Message) []openai.ChatCompletionMessageParamUnion {
	var openaiMessages []openai.ChatCompletionMessageParamUnion

	for _, msg := range messages {
		switch msg.Role {
		case "user":
			openaiMessages = append(openaiMessages, openai.UserMessage(msg.Content))
		case "assistant":
			// For assistant messages, just use the content
			// Tool calls would be handled separately in the API response
			openaiMessages = append(openaiMessages, openai.AssistantMessage(msg.Content))
		case "system":
			openaiMessages = append(openaiMessages, openai.SystemMessage(msg.Content))
		}
	}

	return openaiMessages
}

var _ ports.LLMService = (*OpenAIService)(nil)

// RunToolLoop executes a tool-calling loop with proper OpenAI message sequencing
// Maintains conversation history in OpenAI format internally to ensure tool messages follow tool_calls correctly
func (s *OpenAIService) RunToolLoop(ctx context.Context, question string, tools []domain.Tool, executor domain.ToolExecutor, opts ...domain.Option) (*domain.Response, error) {
	const maxIterations = 5

	cfg := s.applyOptions(opts)

	// Build initial messages in OpenAI format
	chatMessages := s.buildMessages([]string{question}, cfg.SystemPrompt)
	toolParams := s.toolsToOpenAITools(tools)

	// Get initial response
	params := s.buildParams(chatMessages, toolParams, nil, cfg)
	completion, err := s.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, err
	}

	resp := s.completionToResponse(completion)
	var lastChoice *openai.ChatCompletionMessage

	if len(completion.Choices) > 0 {
		lastChoice = &completion.Choices[0].Message
	}

	// Tool loop: while there are tool calls, execute them and get next response
	for iteration := 0; iteration < maxIterations; iteration++ {
		if len(resp.ToolCalls) == 0 {
			break
		}

		// Add the actual assistant message (which includes tool_calls) from the API response
		if lastChoice != nil {
			chatMessages = append(chatMessages, lastChoice.ToParam())
		}

		// Execute all tool calls
		for _, toolCall := range resp.ToolCalls {
			result, err := executor(toolCall.Name, toolCall.Arguments)
			if err != nil {
				result = "Error: " + err.Error()
			}

			// Add tool result message (content first, then tool call ID)
			chatMessages = append(chatMessages, openai.ToolMessage(result, toolCall.ID))
		}

		// Get next response with tool results in context
		params := s.buildParams(chatMessages, toolParams, nil, cfg)
		completion, err := s.client.Chat.Completions.New(ctx, params)
		if err != nil {
			return nil, err
		}

		resp = s.completionToResponse(completion)

		if len(completion.Choices) > 0 {
			lastChoice = &completion.Choices[0].Message
		}
	}

	return resp, nil
}

// RunChatToolLoop executes a tool-calling loop starting from an existing message history
func (s *OpenAIService) RunChatToolLoop(ctx context.Context, messages []domain.Message, tools []domain.Tool, executor domain.ToolExecutor, opts ...domain.Option) (*domain.Response, error) {
	const maxIterations = 5

	cfg := s.applyOptions(opts)

	// Build messages from history
	chatMessages := s.messagesToOpenAIMessages(messages)
	chatMessages = s.prependSystemPrompt(chatMessages, cfg.SystemPrompt)
	toolParams := s.toolsToOpenAITools(tools)

	// Get initial response
	params := s.buildParams(chatMessages, toolParams, nil, cfg)
	completion, err := s.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, err
	}

	resp := s.completionToResponse(completion)
	var lastChoice *openai.ChatCompletionMessage

	if len(completion.Choices) > 0 {
		lastChoice = &completion.Choices[0].Message
	}

	// Tool loop: while there are tool calls, execute them and get next response
	for iteration := 0; iteration < maxIterations; iteration++ {
		if len(resp.ToolCalls) == 0 {
			break
		}

		// Add the actual assistant message (which includes tool_calls) from the API response
		if lastChoice != nil {
			chatMessages = append(chatMessages, lastChoice.ToParam())
		}

		// Execute all tool calls
		for _, toolCall := range resp.ToolCalls {
			result, err := executor(toolCall.Name, toolCall.Arguments)
			if err != nil {
				result = "Error: " + err.Error()
			}

			// Add tool result message (content first, then tool call ID)
			chatMessages = append(chatMessages, openai.ToolMessage(result, toolCall.ID))
		}

		// Get next response with tool results in context
		params := s.buildParams(chatMessages, toolParams, nil, cfg)
		completion, err := s.client.Chat.Completions.New(ctx, params)
		if err != nil {
			return nil, err
		}

		resp = s.completionToResponse(completion)

		if len(completion.Choices) > 0 {
			lastChoice = &completion.Choices[0].Message
		}
	}

	return resp, nil
}
