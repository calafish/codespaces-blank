#!/usr/bin/env python3
"""
Example script for Fish Speech model inference using vLLM.

This script demonstrates how to use the Fish Speech model with vLLM for both
text generation and speech synthesis tasks.
"""

import argparse
import asyncio
from typing import List, Optional

from vllm import LLM, SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid


class FishSpeechInference:
    """Fish Speech model inference wrapper."""
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        dtype: str = "bfloat16",
    ):
        """Initialize Fish Speech inference engine.
        
        Args:
            model_path: Path to the converted Fish Speech model
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
            max_model_len: Maximum model sequence length
            dtype: Model data type
        """
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            trust_remote_code=True,
        )
        
    def generate_text(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> List[str]:
        """Generate text using Fish Speech model.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            List of generated texts
        """
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def generate_speech_tokens(
        self,
        prompts: List[str],
        max_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.9,
        use_codebook_sampling: bool = True,
    ) -> List[List[int]]:
        """Generate speech tokens using Fish Speech model.
        
        Args:
            prompts: List of input text prompts
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            use_codebook_sampling: Whether to use codebook sampling
            
        Returns:
            List of generated speech token sequences
        """
        # For speech generation, we might need special prompts or tokens
        # This is a simplified example - actual implementation would depend
        # on Fish Speech's specific token format
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            # Custom parameters for speech generation
            stop_token_ids=[],  # Define appropriate stop tokens
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Extract token IDs instead of text
        token_sequences = []
        for output in outputs:
            # This would need to be adapted based on how Fish Speech
            # handles speech token generation
            token_ids = output.outputs[0].token_ids
            token_sequences.append(token_ids)
        
        return token_sequences


class AsyncFishSpeechInference:
    """Async Fish Speech model inference wrapper."""
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        dtype: str = "bfloat16",
    ):
        """Initialize async Fish Speech inference engine."""
        self.engine = AsyncLLMEngine.from_engine_args(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            trust_remote_code=True,
        )
    
    async def generate_text_async(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> List[str]:
        """Generate text asynchronously."""
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        
        request_id = random_uuid()
        results = []
        
        for prompt in prompts:
            request_id = random_uuid()
            self.engine.add_request(request_id, prompt, sampling_params)
            
            # Wait for completion
            async for request_output in self.engine.generate():
                if request_output.request_id == request_id:
                    if request_output.finished:
                        results.append(request_output.outputs[0].text)
                        break
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Fish Speech model inference with vLLM"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the converted Fish Speech model"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, I am Fish Speech model.",
        help="Input prompt for generation"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["text", "speech"],
        default="text",
        help="Task type: text generation or speech synthesis"
    )
    parser.add_argument(
        "--async-mode",
        action="store_true",
        help="Use async inference"
    )
    
    args = parser.parse_args()
    
    if args.async_mode:
        # Async inference
        async def run_async():
            inference = AsyncFishSpeechInference(
                model_path=args.model_path,
                tensor_parallel_size=args.tensor_parallel_size,
            )
            
            results = await inference.generate_text_async(
                prompts=[args.prompt],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            
            print("Generated text:")
            for i, result in enumerate(results):
                print(f"{i+1}: {result}")
        
        asyncio.run(run_async())
    
    else:
        # Sync inference
        inference = FishSpeechInference(
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        
        if args.task == "text":
            results = inference.generate_text(
                prompts=[args.prompt],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            
            print("Generated text:")
            for i, result in enumerate(results):
                print(f"{i+1}: {result}")
        
        elif args.task == "speech":
            token_sequences = inference.generate_speech_tokens(
                prompts=[args.prompt],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            
            print("Generated speech tokens:")
            for i, tokens in enumerate(token_sequences):
                print(f"{i+1}: {tokens[:50]}...")  # Show first 50 tokens
                print(f"    Total tokens: {len(tokens)}")


if __name__ == "__main__":
    main()