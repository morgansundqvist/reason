package domain

import "time"

// ========== Core Domain Models ==========

// Message represents a single message in a conversation
type Message struct {
	Role      string // "user", "assistant", "system"
	Content   string
	ToolCalls []ToolCall // Tool calls made by assistant (only for assistant messages)
}

// ParameterType defines the JSON schema type of a tool parameter
type ParameterType string

const (
	TypeString  ParameterType = "string"
	TypeNumber  ParameterType = "number"
	TypeInteger ParameterType = "integer"
	TypeBoolean ParameterType = "boolean"
	TypeArray   ParameterType = "array"
	TypeObject  ParameterType = "object"
)

// Parameter represents a single parameter in a tool's input schema
type Parameter struct {
	Name        string        // Parameter name
	Type        ParameterType // JSON schema type
	Description string        // What this parameter is for
	Required    bool          // Whether this parameter is required
	Enum        []string      // Optional: allowed values for this parameter
}

// Tool defines a tool/function that the LLM can call
type Tool struct {
	Name        string      // e.g., "search_web", "calculate"
	Description string      // What the tool does
	Parameters  []Parameter // Tool input parameters with type information
}

// ToolCall represents a tool invocation by the LLM
type ToolCall struct {
	ID        string                 // Unique identifier for this call
	Name      string                 // Tool name
	Arguments map[string]interface{} // Parsed arguments
}

// Response represents the LLM's response
type Response struct {
	Content    string     // Text response
	ToolCalls  []ToolCall // Tools the LLM wants to call (if any)
	StopReason string     // "end_turn", "tool_use", etc.
	Usage      *Usage
	RawMessage map[string]interface{} // Raw message data for proper API continuation (tool_calls info)
}

// Usage tracks token usage
type Usage struct {
	InputTokens  int
	OutputTokens int
}

// ToolResult represents the result of executing a tool
type ToolResult struct {
	ToolCallID string // ID of the tool call this result responds to
	ToolName   string // Name of the tool that was executed
	Result     string // Result of the tool execution
}

// AssistantResponse wraps the full assistant response needed for tool looping
// It tracks both the content/tool calls AND the raw message structure for API continuations
type AssistantResponse struct {
	Response *Response              // The domain response
	RawJSON  map[string]interface{} // Raw choice message JSON from API for continuation
}

// ToolExecutor is a callback function that executes a tool and returns its result
// It receives the tool name and arguments, and returns a string result or an error
type ToolExecutor func(toolName string, arguments map[string]interface{}) (string, error)

// ========== Configuration Options ==========

// Option is a functional option for configuring LLM calls
type Option func(*CallConfig)

// CallConfig holds configuration for API calls
type CallConfig struct {
	Model        string        // Model name (e.g., "gpt-4", "claude-opus")
	Temperature  float64       // 0.0-1.0, controls randomness
	MaxTokens    int           // Maximum output tokens
	SystemPrompt string        // Custom system prompt override
	StrictJSON   bool          // Enforce JSON output format
	Effort       Effort        // Request intensity level
	RateLimitKey string        // For rate limiting
	Think        bool          // Enable extended thinking (Ollama only)
	Timeout      time.Duration // HTTP client timeout (Ollama only; 0 = use default 2 minutes)
	KeepAlive    *int          // Ollama keep_alive seconds: nil=default, 0=unload, -1=forever, N=seconds
}

// Effort defines request intensity levels
type Effort string

const (
	EffortLow    Effort = "low"
	EffortMedium Effort = "medium"
	EffortHigh   Effort = "high"
)

// Option builders
func WithModel(model string) Option {
	return func(cfg *CallConfig) { cfg.Model = model }
}

func WithTemperature(temp float64) Option {
	return func(cfg *CallConfig) { cfg.Temperature = temp }
}

func WithMaxTokens(tokens int) Option {
	return func(cfg *CallConfig) { cfg.MaxTokens = tokens }
}

func WithSystemPrompt(prompt string) Option {
	return func(cfg *CallConfig) { cfg.SystemPrompt = prompt }
}

func WithStrictJSON(strict bool) Option {
	return func(cfg *CallConfig) { cfg.StrictJSON = strict }
}

func WithEffort(effort Effort) Option {
	return func(cfg *CallConfig) { cfg.Effort = effort }
}

func WithRateLimitKey(key string) Option {
	return func(cfg *CallConfig) { cfg.RateLimitKey = key }
}

func WithThink(think bool) Option {
	return func(cfg *CallConfig) { cfg.Think = think }
}

func WithTimeout(d time.Duration) Option {
	return func(cfg *CallConfig) { cfg.Timeout = d }
}

func WithKeepAlive(seconds int) Option {
	return func(cfg *CallConfig) { cfg.KeepAlive = &seconds }
}
