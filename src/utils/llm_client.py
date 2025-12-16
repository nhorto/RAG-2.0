"""Reusable LLM client with retry logic and error handling."""

import logging
from typing import Optional, Dict, List
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper for OpenAI LLM calls with retry logic and standardized error handling."""

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4",
        temperature: float = 0.3,
        max_tokens: int = 500
    ):
        """Initialize LLM client.

        Args:
            client: OpenAI client instance
            model: Model name (gpt-4, gpt-3.5-turbo, etc.)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(min=1, max=4),
        reraise=True
    )
    def call_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[Dict] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Call LLM with exponential backoff retry logic.

        Args:
            system_prompt: System message content
            user_prompt: User message content
            response_format: Optional response format (e.g., {"type": "json_object"})
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            LLM response content as string

        Raises:
            Exception: If all retry attempts fail
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens
        }

        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
