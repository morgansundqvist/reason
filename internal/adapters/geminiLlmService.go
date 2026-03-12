package adapters

import (
	"context"
	"fmt"
	"strings"

	"github.com/morgansundqvist/reason/internal/domain"
	"github.com/morgansundqvist/reason/internal/ports"
	"google.golang.org/genai"
)

const geminiMaxToolIterations = 5

// GeminiService implements the ports.LLMService interface using Google's Gemini API.
type GeminiService struct {
	client *genai.Client
	config *GeminiConfig
}

// GeminiConfig holds configuration for the Gemini service.
type GeminiConfig struct {
	APIKey string
	Model  string
}

// NewGeminiService creates a new Gemini LLM service.
func NewGeminiService(ctx context.Context, cfg *GeminiConfig) (*GeminiService, error) {
	if cfg == nil {
		return nil, fmt.Errorf("gemini config is required")
	}
	if strings.TrimSpace(cfg.APIKey) == "" {
		return nil, fmt.Errorf("gemini API key is required")
	}

	model := cfg.Model
	if strings.TrimSpace(model) == "" {
		model = "gemini-3.1-flash-lite-preview"
	}

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  cfg.APIKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, fmt.Errorf("create gemini client: %w", err)
	}

	return &GeminiService{
		client: client,
		config: &GeminiConfig{
			APIKey: cfg.APIKey,
			Model:  model,
		},
	}, nil
}

// AskQuestion handles simple text-only questions.
func (s *GeminiService) AskQuestion(ctx context.Context, question string, opts ...domain.Option) (*domain.Response, error) {
	cfg := s.applyOptions(opts)
	contents := s.buildContents([]string{question})
	gcfg := s.buildGeminiConfig(nil, nil, cfg)
	return s.generateContent(ctx, contents, gcfg, cfg)
}

// AskQuestionWithTools allows the LLM to call tools.
func (s *GeminiService) AskQuestionWithTools(ctx context.Context, question string, tools []domain.Tool, opts ...domain.Option) (*domain.Response, error) {
	cfg := s.applyOptions(opts)
	contents := s.buildContents([]string{question})
	gcfg := s.buildGeminiConfig(tools, nil, cfg)
	return s.generateContent(ctx, contents, gcfg, cfg)
}

// AskTypedQuestion asks a question expecting structured output.
func (s *GeminiService) AskTypedQuestion(ctx context.Context, question string, responseSchema map[string]interface{}, opts ...domain.Option) (*domain.Response, error) {
	cfg := s.applyOptions(opts)
	contents := s.buildContents([]string{question})
	gcfg := s.buildGeminiConfig(nil, responseSchema, cfg)
	return s.generateContent(ctx, contents, gcfg, cfg)
}

// AskTypedQuestionWithTools combines typed responses with tool calls.
func (s *GeminiService) AskTypedQuestionWithTools(ctx context.Context, question string, tools []domain.Tool, responseSchema map[string]interface{}, opts ...domain.Option) (*domain.Response, error) {
	cfg := s.applyOptions(opts)
	contents := s.buildContents([]string{question})
	gcfg := s.buildGeminiConfig(tools, responseSchema, cfg)
	return s.generateContent(ctx, contents, gcfg, cfg)
}

// Chat maintains a conversation across multiple turns.
func (s *GeminiService) Chat(ctx context.Context, messages []domain.Message, opts ...domain.Option) (*domain.Response, error) {
	cfg := s.applyOptions(opts)
	contents := s.messagesToGeminiContents(messages)
	gcfg := s.buildGeminiConfig(nil, nil, cfg)
	return s.generateContent(ctx, contents, gcfg, cfg)
}

// ChatWithTools is Chat with tool support.
func (s *GeminiService) ChatWithTools(ctx context.Context, messages []domain.Message, tools []domain.Tool, opts ...domain.Option) (*domain.Response, error) {
	cfg := s.applyOptions(opts)
	contents := s.messagesToGeminiContents(messages)
	gcfg := s.buildGeminiConfig(tools, nil, cfg)
	return s.generateContent(ctx, contents, gcfg, cfg)
}

// RunToolLoop executes a tool-calling loop for an initial user question.
func (s *GeminiService) RunToolLoop(ctx context.Context, question string, tools []domain.Tool, executor domain.ToolExecutor, opts ...domain.Option) (*domain.Response, error) {
	cfg := s.applyOptions(opts)
	contents := s.buildContents([]string{question})
	gcfg := s.buildGeminiConfig(tools, nil, cfg)
	return s.runToolLoop(ctx, contents, gcfg, tools, executor, cfg)
}

// RunChatToolLoop executes a tool-calling loop with an existing message history.
func (s *GeminiService) RunChatToolLoop(ctx context.Context, messages []domain.Message, tools []domain.Tool, executor domain.ToolExecutor, opts ...domain.Option) (*domain.Response, error) {
	cfg := s.applyOptions(opts)
	contents := s.messagesToGeminiContents(messages)
	gcfg := s.buildGeminiConfig(tools, nil, cfg)
	return s.runToolLoop(ctx, contents, gcfg, tools, executor, cfg)
}

func (s *GeminiService) generateContent(ctx context.Context, contents []*genai.Content, gcfg *genai.GenerateContentConfig, cfg *domain.CallConfig) (*domain.Response, error) {
	resp, err := s.client.Models.GenerateContent(ctx, cfg.Model, contents, gcfg)
	if err != nil {
		return nil, fmt.Errorf("gemini generate content: %w", err)
	}
	return s.geminiResponseToResponse(resp), nil
}

func (s *GeminiService) runToolLoop(ctx context.Context, contents []*genai.Content, gcfg *genai.GenerateContentConfig, tools []domain.Tool, executor domain.ToolExecutor, cfg *domain.CallConfig) (*domain.Response, error) {
	resp, err := s.client.Models.GenerateContent(ctx, cfg.Model, contents, gcfg)
	if err != nil {
		return nil, fmt.Errorf("gemini generate content: %w", err)
	}

	domainResp := s.geminiResponseToResponse(resp)

	for iteration := 0; iteration < geminiMaxToolIterations; iteration++ {
		functionCalls := resp.FunctionCalls()
		if len(functionCalls) == 0 {
			break
		}

		// Append the model's response (with function calls) to history
		if len(resp.Candidates) > 0 && resp.Candidates[0].Content != nil {
			contents = append(contents, resp.Candidates[0].Content)
		}

		// Execute each tool call and collect FunctionResponse parts
		var responseParts []*genai.Part
		for _, fc := range functionCalls {
			result, execErr := executor(fc.Name, fc.Args)
			if execErr != nil {
				result = "Error: " + execErr.Error()
			}

			responseParts = append(responseParts, genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
				"output": result,
			}))
		}

		// Append all function responses as a single user-role content
		contents = append(contents, genai.NewContentFromParts(responseParts, "user"))

		// Get next response
		resp, err = s.client.Models.GenerateContent(ctx, cfg.Model, contents, gcfg)
		if err != nil {
			return nil, fmt.Errorf("gemini generate content (tool loop): %w", err)
		}

		domainResp = s.geminiResponseToResponse(resp)
	}

	return domainResp, nil
}

// Helper methods

func (s *GeminiService) applyOptions(opts []domain.Option) *domain.CallConfig {
	cfg := &domain.CallConfig{Model: s.config.Model}
	for _, opt := range opts {
		opt(cfg)
	}
	if cfg.Model == "" {
		cfg.Model = s.config.Model
	}
	return cfg
}

func (s *GeminiService) buildContents(questions []string) []*genai.Content {
	contents := make([]*genai.Content, 0, len(questions))
	for _, q := range questions {
		contents = append(contents, genai.NewContentFromText(q, "user"))
	}
	return contents
}

func (s *GeminiService) buildGeminiConfig(tools []domain.Tool, responseSchema map[string]interface{}, cfg *domain.CallConfig) *genai.GenerateContentConfig {
	gcfg := &genai.GenerateContentConfig{}

	if cfg.SystemPrompt != "" {
		gcfg.SystemInstruction = genai.NewContentFromText(cfg.SystemPrompt, "system")
	}

	if cfg.Temperature > 0 {
		temp := float32(cfg.Temperature)
		gcfg.Temperature = &temp
	}

	if cfg.MaxTokens > 0 {
		gcfg.MaxOutputTokens = int32(cfg.MaxTokens)
	}

	if len(tools) > 0 {
		gcfg.Tools = s.toolsToGeminiTools(tools)
	}

	if responseSchema != nil {
		gcfg.ResponseMIMEType = "application/json"
		gcfg.ResponseJsonSchema = responseSchema
	}

	return gcfg
}

func (s *GeminiService) toolsToGeminiTools(tools []domain.Tool) []*genai.Tool {
	declarations := make([]*genai.FunctionDeclaration, 0, len(tools))

	for _, tool := range tools {
		declarations = append(declarations, &genai.FunctionDeclaration{
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  s.parametersToGeminiSchema(tool.Parameters),
		})
	}

	return []*genai.Tool{{FunctionDeclarations: declarations}}
}

func (s *GeminiService) parametersToGeminiSchema(params []domain.Parameter) *genai.Schema {
	properties := make(map[string]*genai.Schema, len(params))
	required := make([]string, 0, len(params))

	for _, p := range params {
		schema := &genai.Schema{
			Type:        geminiParamType(p.Type),
			Description: p.Description,
		}

		if len(p.Enum) > 0 {
			schema.Enum = p.Enum
		}

		properties[p.Name] = schema

		if p.Required {
			required = append(required, p.Name)
		}
	}

	return &genai.Schema{
		Type:       genai.TypeObject,
		Properties: properties,
		Required:   required,
	}
}

func geminiParamType(t domain.ParameterType) genai.Type {
	switch t {
	case domain.TypeString:
		return genai.TypeString
	case domain.TypeNumber:
		return genai.TypeNumber
	case domain.TypeInteger:
		return genai.TypeInteger
	case domain.TypeBoolean:
		return genai.TypeBoolean
	case domain.TypeArray:
		return genai.TypeArray
	case domain.TypeObject:
		return genai.TypeObject
	default:
		return genai.TypeString
	}
}

func (s *GeminiService) messagesToGeminiContents(messages []domain.Message) []*genai.Content {
	contents := make([]*genai.Content, 0, len(messages))

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			// System messages are handled via SystemInstruction in config; skip here.
			// If no explicit system prompt option is set but a system message exists in history,
			// we still skip it from the contents slice (caller should use WithSystemPrompt).
			continue
		case "user":
			contents = append(contents, genai.NewContentFromText(msg.Content, "user"))
		case "assistant":
			if len(msg.ToolCalls) > 0 {
				// Assistant message with tool calls — use FunctionCall parts
				parts := make([]*genai.Part, 0, len(msg.ToolCalls))
				for _, tc := range msg.ToolCalls {
					parts = append(parts, genai.NewPartFromFunctionCall(tc.Name, tc.Arguments))
				}
				if msg.Content != "" {
					parts = append([]*genai.Part{genai.NewPartFromText(msg.Content)}, parts...)
				}
				contents = append(contents, genai.NewContentFromParts(parts, "model"))
			} else {
				contents = append(contents, genai.NewContentFromText(msg.Content, "model"))
			}
		}
	}

	return contents
}

// geminiResponseText extracts and concatenates all text parts from the first candidate
// without triggering the SDK's warning log that fires when non-text parts (e.g. FunctionCall)
// are present. This is a safe replacement for resp.Text().
func geminiResponseText(resp *genai.GenerateContentResponse) string {
	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return ""
	}
	var sb strings.Builder
	for _, part := range resp.Candidates[0].Content.Parts {
		if part.Text != "" {
			sb.WriteString(part.Text)
		}
	}
	return sb.String()
}

func (s *GeminiService) geminiResponseToResponse(resp *genai.GenerateContentResponse) *domain.Response {
	domainResp := &domain.Response{}

	if resp == nil {
		return domainResp
	}

	domainResp.Content = geminiResponseText(resp)

	if resp.UsageMetadata != nil {
		domainResp.Usage = &domain.Usage{
			InputTokens:  int(resp.UsageMetadata.PromptTokenCount),
			OutputTokens: int(resp.UsageMetadata.CandidatesTokenCount),
		}
	}

	if len(resp.Candidates) > 0 {
		domainResp.StopReason = string(resp.Candidates[0].FinishReason)
	}

	for _, fc := range resp.FunctionCalls() {
		args := make(map[string]interface{}, len(fc.Args))
		for k, v := range fc.Args {
			args[k] = v
		}
		domainResp.ToolCalls = append(domainResp.ToolCalls, domain.ToolCall{
			ID:        fc.ID,
			Name:      fc.Name,
			Arguments: args,
		})
	}

	return domainResp
}

var _ ports.LLMService = (*GeminiService)(nil)
