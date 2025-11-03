import numpy as np
import joblib
from hmmlearn import hmm

class HMMTrainer:
    def __init__(self, model_path="../models/hmm_model.pkl"):
        self.model = joblib.load(model_path)
        self.char_to_int = {c: i for i, c in enumerate(list("abcdefghijklmnopqrstuvwxyz"))}
        self.int_to_char = {i: c for c, i in self.char_to_int.items()}

    def next_letter_probs(self, prefix):
        seq = np.array([[self.char_to_int[c]] for c in prefix if c in self.char_to_int])
        logprob, next_states = self.model.decode(seq, algorithm="viterbi")
        probs = np.exp(self.model.emissionprob_[next_states[-1]])
        return {self.int_to_char[i]: float(probs[i]) for i in range(len(probs))}
