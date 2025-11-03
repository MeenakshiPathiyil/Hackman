# src/utils.py
import numpy as np
import string

ALPHABET = list(string.ascii_lowercase)

def build_state_vector(masked_word: str, guessed_letters: set, lives_left: int, hmm_probs=None):
    """
    Converts environment info into a flat numeric state vector.
    Components:
        - Masked word one-hot (len × 27)
        - Guessed letters binary (26)
        - Lives_left normalized
        - HMM letter probabilities (26)
    """
    max_len = 12  # truncate/pad to fixed size for agent
    masked_word = masked_word[:max_len].ljust(max_len, "#")
    char_index = {c: i for i, c in enumerate(ALPHABET)}

    # 1. masked word one-hot
    one_hot = []
    for ch in masked_word:
        v = np.zeros(len(ALPHABET) + 1)
        if ch in char_index:
            v[char_index[ch]] = 1
        else:  # underscore or padding
            v[-1] = 1
        one_hot.append(v)
    one_hot = np.concatenate(one_hot)

    # 2. guessed vector
    guessed_vec = np.array([1 if c in guessed_letters else 0 for c in ALPHABET])

    # 3. lives
    lives_vec = np.array([lives_left / 6.0])

    # 4. HMM probabilities
    hmm_vec = np.array(hmm_probs) if hmm_probs is not None else np.zeros(len(ALPHABET))

    return np.concatenate([one_hot, guessed_vec, lives_vec, hmm_vec])
