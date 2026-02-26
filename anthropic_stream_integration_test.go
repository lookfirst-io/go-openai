package openai_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"testing"

	openai "github.com/lookfirst-io/go-openai"
	"github.com/lookfirst-io/go-openai/internal/test"
	"github.com/lookfirst-io/go-openai/internal/test/checks"
)

func TestAzureAnthropicStreamWithRealAnthropicFormat(t *testing.T) {
	// Test streaming with actual Anthropic SSE event format
	server := test.NewTestServer()
	ts := server.OpenAITestServer()
	ts.Start()
	defer ts.Close()

	var requestPath string
	server.RegisterHandler("/v1/messages", func(w http.ResponseWriter, r *http.Request) {
		requestPath = r.URL.Path
		w.Header().Set("Content-Type", "text/event-stream")

		// Simulate real Anthropic streaming response format
		events := []string{
			// message_start event
			`data: {"type":"message_start","message":{"id":"msg_01ABC123","type":"message",` +
				`"role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022",` +
				`"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}`,
			// content_block_start event
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			// content_block_delta events
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" from"}}`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" Anthropic"}}`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}}`,
			// content_block_stop event
			`data: {"type":"content_block_stop","index":0}`,
			// message_delta event
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},` +
				`"usage":{"output_tokens":50}}`,
			// message_stop event
			`data: {"type":"message_stop"}`,
		}

		for _, event := range events {
			_, err := w.Write([]byte(event + "\n\n"))
			checks.NoError(t, err, "Write error")
		}

		// Send [DONE]
		_, err := w.Write([]byte("data: [DONE]\n\n"))
		checks.NoError(t, err, "Write error")
	})

	config := openai.DefaultAzureAnthropicConfig(test.GetTestToken(), ts.URL+"/v1")
	client := openai.NewClientWithConfig(config)

	req := openai.ChatCompletionRequest{
		Model: "claude-3-5-sonnet-20241022",
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: "Say hello",
			},
		},
		MaxTokens: 100,
	}

	stream, err := client.CreateChatCompletionStream(context.Background(), req)
	checks.NoError(t, err, "CreateChatCompletionStream error")
	defer stream.Close()

	if requestPath != "/v1/messages" {
		t.Fatalf("Expected request to /v1/messages but got %s", requestPath)
	}

	result := collectAzureAnthropicStream(t, stream)

	// Verify content
	expectedContent := "Hello from Anthropic!"
	if result.content != expectedContent {
		t.Errorf("Expected content '%s', got '%s'", expectedContent, result.content)
	}

	if result.messageID != "msg_01ABC123" {
		t.Errorf("Expected message ID 'msg_01ABC123', got '%s'", result.messageID)
	}

	if result.model != "claude-3-5-sonnet-20241022" {
		t.Errorf("Expected model 'claude-3-5-sonnet-20241022', got '%s'", result.model)
	}

	if !result.hasRole {
		t.Error("Expected to receive assistant role in first chunk")
	}

	if !result.hasEndTurnFinish {
		t.Error("Expected to receive 'end_turn' finish reason")
	}

	if !result.hasStopFinish {
		t.Error("Expected to receive 'stop' finish reason")
	}
}

type azureAnthropicStreamResult struct {
	content          string
	messageID        string
	model            string
	hasRole          bool
	hasEndTurnFinish bool
	hasStopFinish    bool
}

func collectAzureAnthropicStream(
	t *testing.T,
	stream *openai.ChatCompletionStream,
) azureAnthropicStreamResult {
	t.Helper()

	var result azureAnthropicStreamResult

	for {
		response, streamErr := stream.Recv()
		if errors.Is(streamErr, io.EOF) {
			break
		}
		checks.NoError(t, streamErr, "stream.Recv error")

		if response.ID != "" {
			result.messageID = response.ID
		}
		if response.Model != "" {
			result.model = response.Model
		}

		if len(response.Choices) > 0 {
			choice := response.Choices[0]

			if choice.Delta.Role == "assistant" {
				result.hasRole = true
			}

			result.content += choice.Delta.Content

			if choice.FinishReason == "end_turn" {
				result.hasEndTurnFinish = true
			}
			if choice.FinishReason == openai.FinishReasonStop {
				result.hasStopFinish = true
			}
		}
	}

	return result
}

func TestAnthropicAPITypeStreamWithRealFormat(t *testing.T) {
	// Test that APITypeAnthropic also works (not just APITypeAzureAnthropic)
	server := test.NewTestServer()
	ts := server.OpenAITestServer()
	ts.Start()
	defer ts.Close()

	server.RegisterHandler("/v1/messages", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")

		events := []string{
			`data: {"type":"message_start","message":{"id":"msg_test","type":"message",` +
				`"role":"assistant","content":[],"model":"claude-3-opus",` +
				`"usage":{"input_tokens":5,"output_tokens":0}}}`,
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Test"}}`,
			`data: {"type":"content_block_stop","index":0}`,
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}`,
			`data: {"type":"message_stop"}`,
		}

		for _, event := range events {
			_, err := w.Write([]byte(event + "\n\n"))
			checks.NoError(t, err, "Write error")
		}

		_, err := w.Write([]byte("data: [DONE]\n\n"))
		checks.NoError(t, err, "Write error")
	})

	// Use regular Anthropic config (not Azure)
	config := openai.DefaultAnthropicConfig(test.GetTestToken(), ts.URL+"/v1")
	client := openai.NewClientWithConfig(config)

	req := openai.ChatCompletionRequest{
		Model: "claude-3-opus",
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: "Test",
			},
		},
		MaxTokens: 100,
	}

	stream, err := client.CreateChatCompletionStream(context.Background(), req)
	checks.NoError(t, err, "CreateChatCompletionStream error")
	defer stream.Close()

	receivedContent := ""
	for {
		response, streamErr := stream.Recv()
		if errors.Is(streamErr, io.EOF) {
			break
		}
		checks.NoError(t, streamErr, "stream.Recv error")

		if len(response.Choices) > 0 {
			receivedContent += response.Choices[0].Delta.Content
		}
	}

	if receivedContent != "Test" {
		t.Errorf("Expected content 'Test', got '%s'", receivedContent)
	}
}
