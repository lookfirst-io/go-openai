package openai

// common.go defines common types used throughout the OpenAI API.

// Usage Represents the total token usage per request to OpenAI.
type Usage struct {
	PromptTokens            int                      `json:"prompt_tokens"`
	CompletionTokens        int                      `json:"completion_tokens"`
	TotalTokens             int                      `json:"total_tokens"`
	PromptTokensDetails     *PromptTokensDetails     `json:"prompt_tokens_details"`
	CompletionTokensDetails *CompletionTokensDetails `json:"completion_tokens_details"`
}

// CompletionTokensDetails Breakdown of tokens used in a completion.
type CompletionTokensDetails struct {
	AudioTokens              int `json:"audio_tokens"`
	ReasoningTokens          int `json:"reasoning_tokens"`
	AcceptedPredictionTokens int `json:"accepted_prediction_tokens"`
	RejectedPredictionTokens int `json:"rejected_prediction_tokens"`
}

// PromptTokensDetails Breakdown of tokens used in the prompt.
type PromptTokensDetails struct {
	AudioTokens  int `json:"audio_tokens"`
	CachedTokens int `json:"cached_tokens"`
	// CacheReadInputTokens is Anthropic's count of prompt tokens read from the
	// cache, which some providers report next to CachedTokens. Novita's
	// CachedTokens also counts cache writes, so this is the one to use for
	// reads when present. Nil when the provider doesn't send it.
	CacheReadInputTokens *int `json:"cache_read_input_tokens,omitempty"`
	// CacheWriteTokens is the number of prompt tokens written to the cache, as
	// reported by e.g. OpenRouter and Siraya. Nil when not sent.
	CacheWriteTokens *int `json:"cache_write_tokens,omitempty"`
	// CacheCreationInputTokens is Anthropic's name for the same count, as
	// reported by e.g. Novita. Nil when not sent.
	CacheCreationInputTokens *int `json:"cache_creation_input_tokens,omitempty"`
}
