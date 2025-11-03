# src/hmm_oracle.py
import numpy as np
from .oracle import oracle 

class HMMOracle:
    def _init_(self, model_path="models/hmm_model_2.pkl"):
        pass  

    def predict(self, masked_word: str, guessed_letters: set) -> np.ndarray:
        """
        Converts your oracle()'s dict → np.array(26) for RL agent.
        """
        result_dict = oracle(masked_word.lower(), {c.lower() for c in guessed_letters})
        
        probs = np.zeros(26)
        for letter, prob in result_dict.items():
            idx = ord(letter) - ord('a')
            probs[idx] = prob
        
        return probs
