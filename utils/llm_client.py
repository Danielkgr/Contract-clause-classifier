"""
LLM Client module for zero-shot contract clause classification.
Supports multiple providers via unified API.
"""

import time
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM."""
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    error: Optional[str] = None  # set when the call failed; text is then empty


class LLMClient:
    """Client for zero-shot LLM classification with multiple provider support."""
    
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        """Initialize LLM client.
        
        Args:
            provider: LLM provider ('openai', 'anthropic', 'huggingface', etc.)
            model: Model name to use
        """
        self.provider = provider or config.llm.provider
        self.model = model or config.llm.model
        self.api_key = config.llm.api_key
        self.base_url = config.llm.base_url
        self.temperature = config.llm.temperature
        self.max_tokens = config.llm.max_tokens
        
        if self.model not in config.llm.COSTS:
            logger.warning(
                f"No price listed for {self.model}; costs use the default "
                f"{config.llm.DEFAULT_COST_PER_MILLION} USD per million tokens"
            )
        
        # Try to import litellm for unified API
        self.use_litellm = False
        try:
            import litellm
            self.litellm = litellm
            self.use_litellm = True
            logger.info(f"Using LiteLLM for provider: {self.provider}")
        except ImportError:
            logger.warning("LiteLLM not installed, using direct provider APIs")
            if self.provider == "openai":
                try:
                    from openai import OpenAI
                    self.openai_client = OpenAI(api_key=self.api_key)
                except ImportError:
                    logger.error("OpenAI client not installed")
    
    def get_prompt(self, contract_text: str, clause_type: str) -> str:
        """Generate zero-shot prompt for clause classification.
        
        Args:
            contract_text: Full contract text
            clause_type: Type of clause to classify
            
        Returns:
            Formatted prompt string
        """
        prompt = (
            f'Analyze the following contract text and determine if it contains a "{clause_type}" clause.\n\n'
            f"Contract text:\n"
            f'"""\n{contract_text}\n"""\n\n'
            "Respond with ONLY one word: YES or NO.\n\n"
            "Answer:"
        )
        return prompt
    
    def classify_single(self, contract_text: str, clause_type: str) -> LLMResponse:
        """Classify a single contract for a single clause type.
        
        Args:
            contract_text: Contract text to analyze
            clause_type: Type of clause to look for
            
        Returns:
            LLMResponse with classification result
        """
        prompt = self.get_prompt(contract_text, clause_type)
        
        start_time = time.time()
        
        if self.use_litellm:
            response = self._classify_litellm(prompt)
        elif self.provider == "openai":
            response = self._classify_openai(prompt)
        else:
            raise ValueError(f"Provider {self.provider} not supported")
        
        end_time = time.time()
        
        # Calculate cost
        cost_per_million = config.llm.get_cost_per_million(self.model)
        input_cost = (response["input_tokens"] / 1_000_000) * cost_per_million[0]
        output_cost = (response["output_tokens"] / 1_000_000) * cost_per_million[1]
        total_cost = input_cost + output_cost
        
        return LLMResponse(
            text=response["text"] or "",
            input_tokens=response["input_tokens"],
            output_tokens=response["output_tokens"],
            latency_ms=(end_time - start_time) * 1000,
            cost_usd=total_cost,
            error=response.get("error"),
        )
    
    def classify_clause_exists(self, contract_text: str, clause_type: str) -> tuple:
        """Classify if a clause exists and return (is_present, response).
        
        Args:
            contract_text: Contract text
            clause_type: Clause type to classify
            
        Returns:
            Tuple of (is_present: bool, response: LLMResponse)
            
        Raises:
            RuntimeError: If the LLM call failed, so a failure is never read as NO
        """
        response = self.classify_single(contract_text, clause_type)
        if response.error:
            raise RuntimeError(f"LLM call failed: {response.error}")
        is_present = response.text.strip().upper().startswith("YES")
        return is_present, response
    
    def _classify_litellm(self, prompt: str) -> dict:
        """Classify using LiteLLM."""
        try:
            response = self.litellm.completion(
                model=f"{self.provider}/{self.model}",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key,
                base_url=self.base_url
            )
            text = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            return {"text": text, "input_tokens": input_tokens, "output_tokens": output_tokens}
        except Exception as e:
            logger.error(f"LiteLLM error: {e}")
            return {"text": "", "input_tokens": 0, "output_tokens": 0, "error": str(e)}
    
    def _classify_openai(self, prompt: str) -> dict:
        """Classify using OpenAI API directly."""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            text = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            return {"text": text, "input_tokens": input_tokens, "output_tokens": output_tokens}
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return {"text": "", "input_tokens": 0, "output_tokens": 0, "error": str(e)}
    
    def batch_classify(self, contracts: List[str], clause_type: str) -> List[LLMResponse]:
        """Batch classify multiple contracts.
        
        Args:
            contracts: List of contract texts
            clause_type: Clause type to classify
            
        Returns:
            List of LLMResponse objects
        """
        responses = []
        for contract in contracts:
            response = self.classify_single(contract, clause_type)
            responses.append(response)
        return responses


def get_clause_type_examples() -> Dict[str, str]:
    """Get brief description of each clause type for prompts."""
    return {
        "Governing Law": "Which state or country's law governs the contract",
        "Anti-Assignment": "Consent or notice needed before the contract can be assigned",
        "Cap On Liability": "A cap on a party's liability for breach",
        "Uncapped Liability": "Liability left uncapped, for all breaches or a particular kind",
        "Audit Rights": "A right to audit the other party's books, records, or premises",
        "Termination For Convenience": "A right to terminate without cause",
        "Change Of Control": "Rights triggered by a change of control, merger, or asset sale",
        "Exclusivity": "An exclusive dealing commitment with the other party",
        "Non-Compete": "A restriction on competing with the other party",
        "Insurance": "A requirement to maintain insurance",
        "License Grant": "A licence granted by one party to the other",
        "Warranty Duration": "How long a warranty lasts",
    }
