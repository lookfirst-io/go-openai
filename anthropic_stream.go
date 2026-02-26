package openai

// AnthropicStreamEventType represents the type of event in Anthropic streaming.
type AnthropicStreamEventType string

const (
	AnthropicEventMessageStart      AnthropicStreamEventType = "message_start"
	AnthropicEventContentBlockStart AnthropicStreamEventType = "content_block_start"
	AnthropicEventContentBlockDelta AnthropicStreamEventType = "content_block_delta"
	AnthropicEventContentBlockStop  AnthropicStreamEventType = "content_block_stop"
	AnthropicEventMessageDelta      AnthropicStreamEventType = "message_delta"
	AnthropicEventMessageStop       AnthropicStreamEventType = "message_stop"
	AnthropicEventPing              AnthropicStreamEventType = "ping"
	AnthropicEventError             AnthropicStreamEventType = "error"
)

// AnthropicStreamEvent represents a single SSE event from Anthropic API.
type AnthropicStreamEvent struct {
	Type  AnthropicStreamEventType `json:"type"`
	Index int                      `json:"index,omitempty"`

	// For message_start event
	Message *AnthropicMessage `json:"message,omitempty"`

	// For content_block_start event
	ContentBlock *AnthropicContentBlock `json:"content_block,omitempty"`

	// For content_block_delta and message_delta events
	Delta *AnthropicDelta `json:"delta,omitempty"`

	// For message_delta event
	Usage *AnthropicUsageDelta `json:"usage,omitempty"`
}

// AnthropicMessage represents the message structure in message_start event.
type AnthropicMessage struct {
	ID           string                  `json:"id"`
	Type         string                  `json:"type"`
	Role         string                  `json:"role"`
	Content      []AnthropicContentBlock `json:"content"`
	Model        string                  `json:"model"`
	StopReason   string                  `json:"stop_reason,omitempty"`
	StopSequence string                  `json:"stop_sequence,omitempty"`
	Usage        *AnthropicUsage         `json:"usage,omitempty"`
}

// AnthropicContentBlock represents a content block in Anthropic's response.
type AnthropicContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

// AnthropicDelta represents the delta in streaming events.
type AnthropicDelta struct {
	Type         string `json:"type"`
	Text         string `json:"text,omitempty"`
	StopReason   string `json:"stop_reason,omitempty"`
	StopSequence string `json:"stop_sequence,omitempty"`
}

// AnthropicUsage represents token usage information.
type AnthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// AnthropicUsageDelta represents incremental usage in message_delta event.
type AnthropicUsageDelta struct {
	OutputTokens int `json:"output_tokens"`
}

// ConvertToOpenAIStreamResponse converts an Anthropic streaming event to OpenAI format.
//
//nolint:funlen // switch covers all 8 Anthropic event types; inherently verbose
func (e *AnthropicStreamEvent) ConvertToOpenAIStreamResponse(
	messageID string,
	model string,
	created int64,
) ChatCompletionStreamResponse {
	response := ChatCompletionStreamResponse{
		ID:      messageID,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []ChatCompletionStreamChoice{},
	}

	switch e.Type {
	case AnthropicEventMessageStart:
		if e.Message != nil {
			response.ID = e.Message.ID
			response.Model = e.Message.Model
			response.Choices = []ChatCompletionStreamChoice{
				{
					Index: 0,
					Delta: ChatCompletionStreamChoiceDelta{
						Role: e.Message.Role,
					},
					FinishReason: "",
				},
			}
			if e.Message.Usage != nil {
				response.Usage = &Usage{
					PromptTokens:     e.Message.Usage.InputTokens,
					CompletionTokens: e.Message.Usage.OutputTokens,
					TotalTokens:      e.Message.Usage.InputTokens + e.Message.Usage.OutputTokens,
				}
			}
		}

	case AnthropicEventContentBlockStart:
		response.Choices = []ChatCompletionStreamChoice{
			{
				Index:        e.Index,
				Delta:        ChatCompletionStreamChoiceDelta{},
				FinishReason: "",
			},
		}

	case AnthropicEventContentBlockDelta:
		if e.Delta != nil {
			response.Choices = []ChatCompletionStreamChoice{
				{
					Index: e.Index,
					Delta: ChatCompletionStreamChoiceDelta{
						Content: e.Delta.Text,
					},
					FinishReason: "",
				},
			}
		}

	case AnthropicEventContentBlockStop:
		response.Choices = []ChatCompletionStreamChoice{
			{
				Index:        e.Index,
				Delta:        ChatCompletionStreamChoiceDelta{},
				FinishReason: "",
			},
		}

	case AnthropicEventMessageDelta:
		finishReason := FinishReason("")
		if e.Delta != nil && e.Delta.StopReason != "" {
			finishReason = FinishReason(e.Delta.StopReason)
		}

		response.Choices = []ChatCompletionStreamChoice{
			{
				Index:        0,
				Delta:        ChatCompletionStreamChoiceDelta{},
				FinishReason: finishReason,
			},
		}

		if e.Usage != nil {
			response.Usage = &Usage{
				CompletionTokens: e.Usage.OutputTokens,
			}
		}

	case AnthropicEventMessageStop:
		response.Choices = []ChatCompletionStreamChoice{
			{
				Index:        0,
				Delta:        ChatCompletionStreamChoiceDelta{},
				FinishReason: FinishReasonStop,
			},
		}

	case AnthropicEventPing:
		// Ping events are just keep-alive, return empty choices
		response.Choices = []ChatCompletionStreamChoice{}

	case AnthropicEventError:
		// Error events: return empty choices, caller should handle via error accumulator
		response.Choices = []ChatCompletionStreamChoice{}
	}

	return response
}
