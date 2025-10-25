package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/morgansundqvist/reason"
)

func main() {
	// Get API key from environment
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is not set")
	}

	// Initialize client using public API
	client := reason.NewClient(apiKey, reason.WithModel("gpt-5"))

	ctx := context.Background()

	// Test 1: Simple Question
	fmt.Println("=== Test 1: Simple Question ===")
	testSimpleQuestion(ctx, client)

	// Test 2: Question with Tools
	fmt.Println("\n=== Test 2: Question with Tools ===")
	testQuestionWithTools(ctx, client)

	// Test 3: Typed Question (Structured Output)
	fmt.Println("\n=== Test 3: Typed Question (Structured Output) ===")
	testTypedQuestion(ctx, client)

	// Test 4: Conversation
	fmt.Println("\n=== Test 4: Multi-turn Conversation ===")
	testConversation(ctx, client)

	// Test 5: Tool Calling Loop
	fmt.Println("\n=== Test 5: Tool Calling Loop ===")
	testToolCallingLoop(ctx, client)

	fmt.Println("\n✅ All tests completed successfully!")
}

func testSimpleQuestion(ctx context.Context, client *reason.Client) {
	question := "What is the capital of France?"

	resp, err := client.SimpleQuery(ctx, question, reason.WithTemperature(0.7))
	if err != nil {
		log.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Question: %s\n", question)
	fmt.Printf("Answer: %s\n", resp.Content)
	fmt.Printf("Tokens - Input: %d, Output: %d\n", resp.Usage.InputTokens, resp.Usage.OutputTokens)
}

func testQuestionWithTools(ctx context.Context, client *reason.Client) {
	question := "What is 25 + 17?"

	tools := []reason.Tool{
		{
			Name:        "calculator",
			Description: "Performs basic arithmetic operations",
			Parameters: []reason.Parameter{
				{
					Name:        "operation",
					Type:        reason.TypeString,
					Description: "The operation to perform (add, subtract, multiply, divide)",
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

	resp, err := client.QueryWithTools(ctx, question, tools, reason.WithTemperature(0.3))
	if err != nil {
		log.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Question: %s\n", question)
	fmt.Printf("Response: %s\n", resp.Content)

	if len(resp.ToolCalls) > 0 {
		fmt.Println("Tool Calls:")
		for _, tc := range resp.ToolCalls {
			fmt.Printf("  - Tool: %s\n", tc.Name)
			fmt.Printf("    Args: %v\n", tc.Arguments)
		}
	}

	fmt.Printf("Tokens - Input: %d, Output: %d\n", resp.Usage.InputTokens, resp.Usage.OutputTokens)
}

func testTypedQuestion(ctx context.Context, client *reason.Client) {
	question := "Extract information about Paris: name, population, and country"

	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"name": map[string]interface{}{
				"type":        "string",
				"description": "City name",
			},
			"population": map[string]interface{}{
				"type":        "number",
				"description": "Approximate population",
			},
			"country": map[string]interface{}{
				"type":        "string",
				"description": "Country name",
			},
		},
		"required":             []string{"name", "population", "country"},
		"additionalProperties": false,
	}

	resp, err := client.StructuredQuery(ctx, question, schema, reason.WithStrictJSON(true))
	if err != nil {
		log.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Question: %s\n", question)
	fmt.Printf("Response (JSON): %s\n", resp.Content)

	// Try to parse the response into a structured format
	var result map[string]interface{}
	if err := json.Unmarshal([]byte(resp.Content), &result); err == nil {
		fmt.Println("Parsed Response:")
		for key, value := range result {
			fmt.Printf("  %s: %v\n", key, value)
		}
	}

	fmt.Printf("Tokens - Input: %d, Output: %d\n", resp.Usage.InputTokens, resp.Usage.OutputTokens)
}

func testConversation(ctx context.Context, client *reason.Client) {
	fmt.Println("Starting multi-turn conversation...")

	// Initial message
	messages := []reason.Message{
		{
			Role:    "user",
			Content: "What is the largest planet in our solar system?",
		},
	}

	resp, err := client.Chat(ctx, messages, reason.WithTemperature(0.5))
	if err != nil {
		log.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("User: %s\n", messages[0].Content)
	fmt.Printf("Assistant: %s\n", resp.Content)

	// Continue the conversation
	messages = append(messages, reason.Message{
		Role:    "assistant",
		Content: resp.Content,
	})

	messages = append(messages, reason.Message{
		Role:    "user",
		Content: "How far is it from Earth?",
	})

	resp2, err := client.Chat(ctx, messages, reason.WithTemperature(0.5))
	if err != nil {
		log.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("User: How far is it from Earth?\n")
	fmt.Printf("Assistant: %s\n", resp2.Content)
	fmt.Printf("Total Tokens Used - Input: %d, Output: %d\n", resp2.Usage.InputTokens, resp2.Usage.OutputTokens)
}

func testToolCallingLoop(ctx context.Context, client *reason.Client) {
	question := "Calculate: first add 15 and 23, then multiply the result by 2"

	// Define a simple calculator executor
	calculatorExecutor := func(toolName string, args map[string]interface{}) (string, error) {
		if toolName != "calculator" {
			return "", fmt.Errorf("unknown tool: %s", toolName)
		}

		operation, ok := args["operation"].(string)
		if !ok {
			return "", fmt.Errorf("operation must be a string")
		}

		a, ok := args["a"].(float64)
		if !ok {
			return "", fmt.Errorf("a must be a number")
		}

		b, ok := args["b"].(float64)
		if !ok {
			return "", fmt.Errorf("b must be a number")
		}

		var result float64
		switch operation {
		case "add":
			result = a + b
		case "subtract":
			result = a - b
		case "multiply":
			result = a * b
		case "divide":
			if b == 0 {
				return "", fmt.Errorf("division by zero")
			}
			result = a / b
		default:
			return "", fmt.Errorf("unknown operation: %s", operation)
		}

		return fmt.Sprintf("Result: %v", result), nil
	}

	// Define calculator tool
	tools := []reason.Tool{
		{
			Name:        "calculator",
			Description: "Performs basic arithmetic operations",
			Parameters: []reason.Parameter{
				{
					Name:        "operation",
					Type:        reason.TypeString,
					Description: "The operation to perform (add, subtract, multiply, divide)",
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

	resp, err := client.QueryWithToolLoop(ctx, question, tools, calculatorExecutor, reason.WithTemperature(0.3))
	if err != nil {
		log.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Question: %s\n", question)
	fmt.Printf("Final Response: %s\n", resp.Content)
	fmt.Printf("Tokens - Input: %d, Output: %d\n", resp.Usage.InputTokens, resp.Usage.OutputTokens)
}
