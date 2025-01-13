# Standard library imports
from __future__ import annotations
import json
import os
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

# Third-party imports
import aiohttp
import anthropic

# Local imports
from utils.schemas import (
    BaseProviderConfig,
    OllamaConfig,
    OpenAIConfig,
    AnthropicConfig,
    ChatMessage,
    ChatOptions,
    Provider,
)
from utils.schemas import ProviderFactory


# Abstract base class
class BaseLLM(ABC):
    """Abstract base class for LLM providers"""

    _shared_memory = []

    def __init__(self, config: BaseProviderConfig):
        self.config = config

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert LLM instance to dictionary"""
        pass

    @abstractmethod
    async def chat(
        self, messages: List[ChatMessage], options: Optional[ChatOptions] = None
    ) -> str:
        """Generate a chat completion from the LLM"""
        pass


# Factory class


class LLMFactory:
    """Factory class to create LLM instances"""

    @staticmethod
    def create_llm(provider: Provider, model_name: str = None, **kwargs) -> BaseLLM:
        """
        Create an LLM instance based on the provider

        Args:
            provider: The LLM provider ('ollama', 'openai', or 'anthropic')
            model_name: Name of the model to use
            **kwargs: Additional arguments (api_key, temperature, max_tokens, etc.)
        """
        # Create config dict with provided arguments
        load_dotenv()
        model = model_name or os.getenv(f"{provider.name}_DEFAULT_MODEL")
        config_dict = {"provider": provider.value, "model_name": model, **kwargs}

        config = ProviderFactory.create_config(config_dict)

        # Create LLM instance based on provider
        if provider.lower() == "ollama":
            return OllamaLLM(config)
        elif provider.lower() == "openai":
            return OpenAILLM(config)
        elif provider.lower() == "anthropic":
            return AnthropicLLM(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


# Provider implementations


class OllamaLLM(BaseLLM):
    """Ollama LLM provider implementation"""

    def __init__(self, config: OllamaConfig):
        super().__init__(config)
        self.base_url = config.base_url

    def to_dict(self):
        return {
            "type": "Ollama",
            "model_name": self.config.model_name,
            "base_url": self.base_url,
        }

    async def chat(
        self, messages: List[ChatMessage], options: Optional[ChatOptions] = None
    ) -> str:
        """Generate chat completion using Ollama"""
        async with aiohttp.ClientSession() as session:
            # Convert messages to Ollama format
            formatted_messages = [
                {"role": msg.role.value, "content": msg.content} for msg in messages
            ]

            payload = {
                "model": self.config.model_name,
                "messages": formatted_messages,
                "stream": False,
                "temperature": (
                    options["temperature"] if options else self.config.temperature
                ),
            }

            if options and options.max_tokens:
                payload["max_tokens"] = options.max_tokens

            print(f"Ollama Config: {self.to_dict()}")
            async with session.post(
                f"{self.base_url}/api/chat", json=payload
            ) as response:
                if response.status == 200:
                    try:
                        raw_response = await response.read()
                        response_data = json.loads(raw_response.decode("utf-8"))
                        total_duration_seconds = (
                            response_data.get("total_duration", 0) / 1_000_000_000
                        )
                        print(f"Total duration in seconds: {total_duration_seconds}")
                        return {
                            "message": response_data.get("message", {}),
                            "metadata": {
                                k: v for k, v in response_data.items() if k != "message"
                            },
                        }
                    except (
                        json.JSONDecodeError,
                        AttributeError,
                        UnicodeDecodeError,
                    ) as e:
                        raise ConnectionError(
                            f"Failed to parse Ollama response: {str(e)}"
                        )
                raise ConnectionError(f"Ollama API error: {response.status}")


class OpenAILLM(BaseLLM):
    """OpenAI LLM provider implementation"""

    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def to_dict(self):
        return {
            "type": "OpenAI",
            "model_name": self.config.model_name,
            "base_url": self.base_url,
        }

    async def chat(self, prompt: str) -> str:
        """Generate chat completion using OpenAI"""
        pass


class AnthropicLLM(BaseLLM):
    """Anthropic LLM provider implementation"""

    def __init__(self, config: AnthropicConfig):
        super().__init__(config)
        if not config.api_key:
            config.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not config.api_key:
            raise ValueError("Anthropic API key not found in config or environment")

        self.client = anthropic.Anthropic(api_key=config.api_key)

    def to_dict(self):
        return {"type": "Anthropic", "model_name": self.config.model_name}

    async def chat(
        self, messages: List[ChatMessage], options: Optional[ChatOptions] = None
    ) -> Dict[str, Any]:
        """Generate text response using Anthropic"""
        try:
            # Extract system message if present
            system_message = next(
                (msg.content for msg in messages if msg.role.value == "system"), None
            )

            # Filter out system message and convert other messages to Anthropic format
            formatted_messages = [
                {
                    "role": "assistant" if msg.role.value == "assistant" else "user",
                    "content": msg.content,
                }
                for msg in messages
                if msg.role.value != "system"
            ]

            message = self.client.messages.create(
                model=self.config.model_name,
                # TODO: Add support for additional options
                max_tokens=options["max_tokens"] if options else self.config.max_tokens,
                # temperature=(
                #     options["temperature"] if options else self.config.temperature
                # ),
                messages=formatted_messages,
                system=system_message,
            )

            content = message.content[0].text if message.content else ""

            metadata = {
                "usage": message.usage,
                "model": message.model,
                "id": message.id,
                "type": message.type,
                "role": message.role,
                "stop_reason": message.stop_reason,
                "stop_sequence": message.stop_sequence,
            }

            return {"message": {"content": content}, "metadata": metadata}
        except Exception as e:
            raise RuntimeError(f"Unexpected error with Anthropic API: {str(e)}")
