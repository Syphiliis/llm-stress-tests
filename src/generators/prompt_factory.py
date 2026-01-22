import random
import string
from typing import List

class PromptFactory:
    def __init__(self, seed: int, min_tokens: int, max_tokens: int, prefix: str = ""):
        self._seed = seed
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens
        self._prefix = prefix
        self._rng = random.Random(seed)
        
        # Pre-compute a pool of words to generate nonsense from
        self._word_pool = [
            ''.join(self._rng.choices(string.ascii_lowercase, k=self._rng.randint(3, 10)))
            for _ in range(1000)
        ]

    def generate_prompt(self, tokens_override: int = None) -> str:
        """
        Generates a deterministic random prompt to avoid KV cache hits.
        The length varies between min_tokens and max_tokens (approximate, based on word count).
        """
        _, text = self.generate_prompt_with_size(tokens_override)
        return text

    def generate_prompt_with_size(self, tokens_override: int = None):
        """
        Returns (input_token_count, prompt_text)
        """
        if tokens_override is None:
            num_words = self._rng.randint(self._min_tokens, self._max_tokens)
        else:
            num_words = max(self._min_tokens, min(tokens_override, self._max_tokens))
        
        words = self._rng.choices(self._word_pool, k=num_words)
        nonsense = " ".join(words)
        
        full_prompt = f"{self._prefix}{nonsense}"
        return num_words, full_prompt

    def generate_prompts(self, count: int) -> List[str]:
        return [self.generate_prompt() for _ in range(count)]
