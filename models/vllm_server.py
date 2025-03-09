"""
LLM Inference Client Module

This module provides a client for interacting with local LLM inference servers
using the OpenAI-compatible API format. It supports both chat completions and
text completions modes.
"""

from typing import List, Tuple, Optional, Dict, Any, Union
import logging
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default model path - consider moving this to a config file or environment variable
DEFAULT_MODEL_PATH = "YOUR_DEFAULT_MODEL_PATH"


class LLMClient:
    """Client for interacting with OpenAI-compatible LLM inference servers."""

    def __init__(
        self,
        chat_api_base_url: str = "http://localhost:8090/v1",
        completion_api_base_url: str = "http://localhost:8080/v1",
        api_key: str = "EMPTY"
    ):
        """
        Initialize the LLM client.

        Args:
            chat_api_base_url (str): Base URL for chat completions API.
            completion_api_base_url (str): Base URL for text completions API.
            api_key (str): API key for authentication (may be dummy value for local servers).
        """
        self.chat_api_base_url = chat_api_base_url
        self.completion_api_base_url = completion_api_base_url
        self.api_key = api_key
        
    def create_chat_completion(
        self,
        message: str,
        model: str = DEFAULT_MODEL_PATH,
        temperature: float = 0.7,
        top_p: float = 0.8,
        max_tokens: Optional[int] = 4096,
        repetition_penalty: float = 1.05,
        n: int = 1
    ) -> List[str]:
        """
        Generate chat completions using the chat API.

        Args:
            message (str): User message to generate a response for.
            model (str): Path or name of the model to use.
            temperature (float): Sampling temperature (0.0 to 1.0).
            top_p (float): Nucleus sampling parameter (0.0 to 1.0).
            max_tokens (int, optional): Maximum tokens in the generated response.
            repetition_penalty (float): Penalty for token repetition.
            n (int): Number of completions to generate.

        Returns:
            List[str]: List of generated text completions.
        """
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.chat_api_base_url,
        )
        
        messages = [
            {"role": "user", "content": message}
        ]
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                # Note: max_tokens is commented out in original code
                # max_tokens=max_tokens,
                extra_body={
                    "repetition_penalty": repetition_penalty,
                },
                n=n
            )
            
            texts = [choice.message.content for choice in response.choices]
            return texts
            
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            return []

    def create_text_completion(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL_PATH,
        temperature: float = 0.7,
        top_p: float = 0.8,
        max_tokens: int = 8192,
        repetition_penalty: float = 1.05,
        n: int = 1,
        stop_tokens: List[str] = []
    ) -> Tuple[List[str], List[str]]:
        """
        Generate text completions using the completions API.

        Args:
            prompt (str): Text prompt to complete.
            model (str): Path or name of the model to use.
            temperature (float): Sampling temperature (0.0 to 1.0).
            top_p (float): Nucleus sampling parameter (0.0 to 1.0).
            max_tokens (int): Maximum tokens in the generated response.
            repetition_penalty (float): Penalty for token repetition.
            n (int): Number of completions to generate.
            stop_tokens (List[str]): List of sequences that stop generation.

        Returns:
            Tuple[List[str], List[str]]: Tuple containing lists of generated texts and stop reasons.
        """
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.completion_api_base_url,
        )
        
        try:
            response = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body={
                    "repetition_penalty": repetition_penalty,
                },
                n=n,
                stop=stop_tokens
            )
            
            texts = [choice.text for choice in response.choices]
            stop_reasons = [choice.finish_reason for choice in response.choices]
            
            return texts, stop_reasons
            
        except Exception as e:
            logger.error(f"Text completion failed: {e}")
            return [], []


# Compatibility functions for backward compatibility
def generate_chat_completion(
    message: str,
    model: str = DEFAULT_MODEL_PATH,
    temperature: float = 0.7,
    top_p: float = 0.8,
    max_tokens: int = 4096,
    repetition_penalty: float = 1.05,
    n: int = 1
) -> List[str]:
    """
    Compatibility function for chat completions.
    
    Args:
        message (str): User message to generate a response for.
        model (str): Path or name of the model to use.
        temperature (float): Sampling temperature (0.0 to 1.0).
        top_p (float): Nucleus sampling parameter (0.0 to 1.0).
        max_tokens (int): Maximum tokens in the generated response.
        repetition_penalty (float): Penalty for token repetition.
        n (int): Number of completions to generate.
    
    Returns:
        List[str]: List of generated text completions.
    """
    client = LLMClient()
    return client.create_chat_completion(
        message=message,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        n=n
    )


def generate_completion(
    prompt: str,
    model: str = DEFAULT_MODEL_PATH,
    temperature: float = 0.7,
    top_p: float = 0.8,
    max_tokens: int = 8192,
    repetition_penalty: float = 1.05,
    n: int = 1,
    stop_tokens: List[str] = []
) -> Tuple[List[str], List[str]]:
    """
    Compatibility function for text completions.
    
    Args:
        prompt (str): Text prompt to complete.
        model (str): Path or name of the model to use.
        temperature (float): Sampling temperature (0.0 to 1.0).
        top_p (float): Nucleus sampling parameter (0.0 to 1.0).
        max_tokens (int): Maximum tokens in the generated response.
        repetition_penalty (float): Penalty for token repetition.
        n (int): Number of completions to generate.
        stop_tokens (List[str]): List of sequences that stop generation.
    
    Returns:
        Tuple[List[str], List[str]]: Tuple containing lists of generated texts and stop reasons.
    """
    client = LLMClient()
    return client.create_text_completion(
        prompt=prompt,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        n=n,
        stop_tokens=stop_tokens
    )


if __name__ == "__main__":
    # Example usage
    texts, stop_reasons = generate_completion("What is the capital of France?")