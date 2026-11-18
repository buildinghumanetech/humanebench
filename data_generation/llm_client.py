"""
LLM client with fallback support between OpenRouter and Cerebras APIs.

OpenRouter provides access to multiple providers (Anthropic, xAI, etc.) with optional web search.
Cerebras direct API serves as fallback for reliability but without web search capabilities.
"""
# Paper: utility -- no paper-facing output; provides the chat-completion client, with provider fallback, used by the generation and validation stages of the scenario construction pipeline.

import os
from typing import Optional, Dict, Any, List
from openai import OpenAI


class FallbackLLMClient:
    def __init__(self, openrouter_api_key: Optional[str] = None, cerebras_api_key: Optional[str] = None):
        """Initialize client with both API keys."""
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.cerebras_api_key = cerebras_api_key or os.getenv("CEREBRAS_API_KEY")

        # Initialize OpenRouter client
        if self.openrouter_api_key:
            self.openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key
            )
        else:
            self.openrouter_client = None

        # Initialize Cerebras client
        if self.cerebras_api_key:
            try:
                from cerebras.cloud.sdk import Cerebras
                self.cerebras_client = Cerebras(api_key=self.cerebras_api_key)
            except ImportError:
                print("⚠️  Cerebras SDK not installed. Install with: pip install cerebras-cloud-sdk")
                self.cerebras_client = None
        else:
            self.cerebras_client = None

    def chat_completion(self,
                       openrouter_model: str,
                       cerebras_model: str,
                       messages: List[Dict[str, str]],
                       temperature: float = 0.7,
                       max_tokens: int = 4000,
                       **kwargs) -> Any:
        """
        Try OpenRouter first, fallback to Cerebras if it fails.

        Args:
            openrouter_model: Model ID for OpenRouter
            cerebras_model: Model ID for Cerebras
            messages: Chat messages
            temperature: Temperature setting
            max_tokens: Max tokens to generate
            **kwargs: Additional parameters

        Returns:
            API response object
        """

        # Try OpenRouter first
        if self.openrouter_client:
            try:
                print(f"🔄 Trying OpenRouter with {openrouter_model}...")
                response = self.openrouter_client.chat.completions.create(
                    model=openrouter_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                print(f"✅ OpenRouter successful")
                return response

            except Exception as e:
                print(f"❌ OpenRouter failed: {e}")
                print(f"🔄 Falling back to Cerebras direct API...")
                print(f"⚠️  Note: Web search capabilities will be unavailable via Cerebras")

        # Fallback to Cerebras
        if self.cerebras_client:
            try:
                print(f"🔄 Trying Cerebras direct with {cerebras_model}...")
                response = self.cerebras_client.chat.completions.create(
                    model=cerebras_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                print(f"✅ Cerebras direct successful")
                return response

            except Exception as e:
                print(f"❌ Cerebras direct failed: {e}")
                raise Exception(f"Both OpenRouter and Cerebras APIs failed. OpenRouter: {e}")

        # No working clients
        raise Exception("No working API clients available. Check your API keys and installations.")

    def get_available_apis(self) -> List[str]:
        """Get list of available APIs."""
        available = []
        if self.openrouter_client and self.openrouter_api_key:
            available.append("OpenRouter")
        if self.cerebras_client and self.cerebras_api_key:
            available.append("Cerebras")
        return available

    def health_check(self) -> Dict[str, bool]:
        """Check health of both APIs."""
        health = {}

        # Test OpenRouter
        if self.openrouter_client:
            try:
                # Simple test call
                test_response = self.openrouter_client.chat.completions.create(
                    model="meta-llama/llama-3.3-70b-instruct:free",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5
                )
                health["openrouter"] = True
            except:
                health["openrouter"] = False
        else:
            health["openrouter"] = False

        # Test Cerebras
        if self.cerebras_client:
            try:
                # Simple test call
                test_response = self.cerebras_client.chat.completions.create(
                    model="llama3.1-8b",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5
                )
                health["cerebras"] = True
            except:
                health["cerebras"] = False
        else:
            health["cerebras"] = False

        return health