package openai_test

import (
	"encoding/json"
	"testing"

	openai "github.com/lookfirst-io/go-openai"
)

func TestAnthropicStreamEventParsing(t *testing.T) {
	// Test parsing content_block_delta event (the format you provided)
	jsonData := `{
		"type": "content_block_delta",
		"index": 0,
		"delta": {
			"type": "text_delta",
			"text": " attractions in Seattle:\n\n1. **"
		}
	}`

	var event openai.AnthropicStreamEvent
	err := json.Unmarshal([]byte(jsonData), &event)
	if err != nil {
		t.Fatalf("Failed to parse Anthropic event: %v", err)
	}

	if event.Type != openai.AnthropicEventContentBlockDelta {
		t.Errorf("Expected type %s, got %s", openai.AnthropicEventContentBlockDelta, event.Type)
	}

	if event.Index != 0 {
		t.Errorf("Expected index 0, got %d", event.Index)
	}

	if event.Delta == nil {
		t.Fatal("Delta should not be nil")
	}

	expectedText := " attractions in Seattle:\n\n1. **"
	if event.Delta.Text != expectedText {
		t.Errorf("Expected text %q, got %q", expectedText, event.Delta.Text)
	}
}

func TestAnthropicToOpenAIConversion(t *testing.T) {
	t.Run("message_start", testAnthropicMessageStart)
	t.Run("content_block_delta", testAnthropicContentBlockDelta)
	t.Run("message_delta", testAnthropicMessageDelta)
	t.Run("message_stop", testAnthropicMessageStop)
}

func testAnthropicMessageStart(t *testing.T) {
	t.Helper()

	event := openai.AnthropicStreamEvent{
		Type: openai.AnthropicEventMessageStart,
		Message: &openai.AnthropicMessage{
			ID:    "msg_123",
			Type:  "message",
			Role:  "assistant",
			Model: "claude-3-opus-20240229",
			Usage: &openai.AnthropicUsage{
				InputTokens:  10,
				OutputTokens: 0,
			},
		},
	}

	response := event.ConvertToOpenAIStreamResponse("msg_123", "claude-3-opus-20240229", 1234567890)

	if response.ID != "msg_123" {
		t.Errorf("Expected ID msg_123, got %s", response.ID)
	}

	if response.Model != "claude-3-opus-20240229" {
		t.Errorf("Expected model claude-3-opus-20240229, got %s", response.Model)
	}

	if len(response.Choices) != 1 {
		t.Fatalf("Expected 1 choice, got %d", len(response.Choices))
	}

	if response.Choices[0].Delta.Role != "assistant" {
		t.Errorf("Expected role assistant, got %s", response.Choices[0].Delta.Role)
	}
}

func testAnthropicContentBlockDelta(t *testing.T) {
	t.Helper()

	event := openai.AnthropicStreamEvent{
		Type:  openai.AnthropicEventContentBlockDelta,
		Index: 0,
		Delta: &openai.AnthropicDelta{
			Type: "text_delta",
			Text: " attractions in Seattle:\n\n1. **",
		},
	}

	response := event.ConvertToOpenAIStreamResponse("msg_123", "claude-3-opus-20240229", 1234567890)

	if len(response.Choices) != 1 {
		t.Fatalf("Expected 1 choice, got %d", len(response.Choices))
	}

	expectedText := " attractions in Seattle:\n\n1. **"
	if response.Choices[0].Delta.Content != expectedText {
		t.Errorf("Expected content %q, got %q", expectedText, response.Choices[0].Delta.Content)
	}

	if response.Choices[0].Index != 0 {
		t.Errorf("Expected index 0, got %d", response.Choices[0].Index)
	}
}

func testAnthropicMessageDelta(t *testing.T) {
	t.Helper()

	event := openai.AnthropicStreamEvent{
		Type: openai.AnthropicEventMessageDelta,
		Delta: &openai.AnthropicDelta{
			StopReason: "end_turn",
		},
		Usage: &openai.AnthropicUsageDelta{
			OutputTokens: 150,
		},
	}

	response := event.ConvertToOpenAIStreamResponse("msg_123", "claude-3-opus-20240229", 1234567890)

	if len(response.Choices) != 1 {
		t.Fatalf("Expected 1 choice, got %d", len(response.Choices))
	}

	if response.Choices[0].FinishReason != "end_turn" {
		t.Errorf("Expected finish_reason end_turn, got %s", response.Choices[0].FinishReason)
	}

	if response.Usage == nil {
		t.Fatal("Usage should not be nil")
	}

	if response.Usage.CompletionTokens != 150 {
		t.Errorf("Expected 150 completion tokens, got %d", response.Usage.CompletionTokens)
	}
}

func testAnthropicMessageStop(t *testing.T) {
	t.Helper()

	event := openai.AnthropicStreamEvent{
		Type: openai.AnthropicEventMessageStop,
	}

	response := event.ConvertToOpenAIStreamResponse("msg_123", "claude-3-opus-20240229", 1234567890)

	if len(response.Choices) != 1 {
		t.Fatalf("Expected 1 choice, got %d", len(response.Choices))
	}

	if response.Choices[0].FinishReason != openai.FinishReasonStop {
		t.Errorf("Expected finish_reason stop, got %s", response.Choices[0].FinishReason)
	}
}
