package openai

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"time"

	utils "github.com/lookfirst-io/go-openai/internal"
)

var (
	headerData  = regexp.MustCompile(`^data:\s*`)
	errorPrefix = regexp.MustCompile(`^data:\s*{"error":`)
)

type streamable interface {
	ChatCompletionStreamResponse | CompletionResponse
}

type streamReader[T streamable] struct {
	emptyMessagesLimit uint
	isFinished         bool

	reader         *bufio.Reader
	response       *http.Response
	errAccumulator utils.ErrorAccumulator
	unmarshaler    utils.Unmarshaler

	httpHeader
	
	// API type for handling different response formats
	apiType APIType
	
	// For Anthropic stream conversion
	anthropicMessageID string
	anthropicModel     string
	anthropicCreated   int64
}

func (stream *streamReader[T]) Recv() (response T, err error) {
	rawLine, err := stream.RecvRaw()
	if err != nil {
		return
	}

	// Check if we need to convert from Anthropic format
	if stream.apiType == APITypeAzureAnthropic || stream.apiType == APITypeAnthropic {
		// Try to parse as Anthropic event first
		var anthropicEvent AnthropicStreamEvent
		if parseErr := stream.unmarshaler.Unmarshal(rawLine, &anthropicEvent); parseErr == nil && anthropicEvent.Type != "" {
			// Successfully parsed as Anthropic event with valid type, convert to OpenAI format
			converted := stream.convertAnthropicEvent(&anthropicEvent)
			
			// Type assert and return
			if openAIResp, ok := any(converted).(T); ok {
				return openAIResp, nil
			}
		}
		// If parsing as Anthropic failed or type is empty, fall through to OpenAI parsing
	}

	// Default: parse as OpenAI format
	err = stream.unmarshaler.Unmarshal(rawLine, &response)
	if err != nil {
		return
	}
	return response, nil
}

func (stream *streamReader[T]) convertAnthropicEvent(event *AnthropicStreamEvent) T {
	var zero T
	
	// Update state from message_start event
	if event.Type == AnthropicEventMessageStart && event.Message != nil {
		stream.anthropicMessageID = event.Message.ID
		stream.anthropicModel = event.Message.Model
		// Use current Unix timestamp
		if stream.anthropicCreated == 0 {
			stream.anthropicCreated = getCurrentUnixTime()
		}
	}
	
	// Ensure we have a timestamp set
	if stream.anthropicCreated == 0 {
		stream.anthropicCreated = getCurrentUnixTime()
	}
	
	// Convert the event to OpenAI format
	converted := event.ConvertToOpenAIStreamResponse(
		stream.anthropicMessageID,
		stream.anthropicModel,
		stream.anthropicCreated,
	)
	
	// Type assert and return
	if result, ok := any(converted).(T); ok {
		return result
	}
	
	return zero
}

func getCurrentUnixTime() int64 {
	return time.Now().Unix()
}

func (stream *streamReader[T]) RecvRaw() ([]byte, error) {
	if stream.isFinished {
		return nil, io.EOF
	}

	return stream.processLines()
}

//nolint:gocognit
func (stream *streamReader[T]) processLines() ([]byte, error) {
	var (
		emptyMessagesCount uint
		hasErrorPrefix     bool
	)

	for {
		rawLine, readErr := stream.reader.ReadBytes('\n')
		if readErr != nil || hasErrorPrefix {
			respErr := stream.unmarshalError()
			if respErr != nil {
				return nil, fmt.Errorf("error, %w", respErr.Error)
			}
			return nil, readErr
		}

		noSpaceLine := bytes.TrimSpace(rawLine)
		if errorPrefix.Match(noSpaceLine) {
			hasErrorPrefix = true
		}
		if !headerData.Match(noSpaceLine) || hasErrorPrefix {
			if hasErrorPrefix {
				noSpaceLine = headerData.ReplaceAll(noSpaceLine, nil)
			}
			writeErr := stream.errAccumulator.Write(noSpaceLine)
			if writeErr != nil {
				return nil, writeErr
			}
			emptyMessagesCount++
			if emptyMessagesCount > stream.emptyMessagesLimit {
				return nil, ErrTooManyEmptyStreamMessages
			}

			continue
		}

		noPrefixLine := headerData.ReplaceAll(noSpaceLine, nil)
		if string(noPrefixLine) == "[DONE]" {
			stream.isFinished = true
			return nil, io.EOF
		}

		return noPrefixLine, nil
	}
}

func (stream *streamReader[T]) unmarshalError() (errResp *ErrorResponse) {
	errBytes := stream.errAccumulator.Bytes()
	if len(errBytes) == 0 {
		return
	}

	err := stream.unmarshaler.Unmarshal(errBytes, &errResp)
	if err != nil {
		errResp = nil
	}

	return
}

func (stream *streamReader[T]) Close() error {
	return stream.response.Body.Close()
}
