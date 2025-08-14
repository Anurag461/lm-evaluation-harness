import torch
from typing import List, Tuple, Union, Optional, Dict
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model


@register_model("complement")
class ComplementModel(TemplateLM):
    """
    A model that always returns "C" for every prompt.
    Likelihood is 1.0 for "C" and 0.0 for all other outputs.
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        self._max_length = 2048
        self._max_gen_toks = 256
        
    @property
    def eot_token_id(self):
        return 0  # Dummy token ID
        
    @property
    def max_length(self):
        return self._max_length
        
    @property
    def max_gen_toks(self):
        return self._max_gen_toks
        
    @property
    def tokenizer_name(self) -> str:
        return "complement_tokenizer"
        
    def tok_encode(
        self,
        string: Union[str, List[str]],
        left_truncate_len: int = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
    ) -> Union[List[int], List[List[int]]]:
        """
        Simple tokenization that maps each character to its ASCII value.
        """
        if isinstance(string, str):
            tokens = [ord(c) for c in string]
            if left_truncate_len:
                tokens = tokens[-left_truncate_len:]
            return tokens
        else:
            # Handle list of strings
            all_tokens = []
            for s in string:
                tokens = [ord(c) for c in s]
                if left_truncate_len:
                    tokens = tokens[-left_truncate_len:]
                all_tokens.append(tokens)
            return all_tokens
    
    def tok_decode(self, tokens: List[int]) -> str:
        """
        Decode tokens back to string.
        """
        try:
            return ''.join(chr(token) for token in tokens if 0 <= token <= 127)
        except:
            return "C"
    
    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        """
        Always return "C" for every request.
        """
        return ["C"] * len(requests)
    
    def loglikelihood(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        """
        Return likelihood 1.0 for "C", 0.0 for everything else.
        """
        results = []
        for request in requests:
            context, continuation = request.args
            if continuation.strip() == "C":
                # High likelihood for "C"
                results.append((0.0, True))  # log(1.0) = 0.0, is_greedy = True
            else:
                # Very low likelihood for anything else
                results.append((-float('inf'), False))  # log(0.0) = -inf, is_greedy = False
        return results
    
    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        """
        For rolling loglikelihood, return 0.0 (log probability of 1.0) for all requests.
        """
        return [0.0] * len(requests)
    
    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        """
        Token-level loglikelihood computation.
        """
        results = []
        for request in requests:
            cache_key, context_tokens, continuation_tokens = request
            # Decode continuation tokens to check if it's "C"
            continuation_text = self.tok_decode(continuation_tokens)
            
            if continuation_text.strip() == "C":
                results.append((0.0, True))  # log(1.0) = 0.0
            else:
                results.append((-float('inf'), False))  # log(0.0) = -inf 
