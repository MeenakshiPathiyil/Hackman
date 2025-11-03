import numpy as np
import os
import joblib
from scipy.special import logsumexp

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "hmm_model.pkl")

# Load model
obj = joblib.load(model_path)
model = obj["model"]
char_to_int = obj["char_to_int"]
int_to_char = obj["int_to_char"]

# Extract HMM parameters
N = model.n_components
V = model.emissionprob_.shape[1]
log_start = np.log(model.startprob_ + 1e-12)
log_trans = np.log(model.transmat_ + 1e-12)
log_emit = np.log(model.emissionprob_ + 1e-12)

# Indexes for all lowercase letters
letter_indexes = [char_to_int[c] for c in list("abcdefghijklmnopqrstuvwxyz")]

def pattern_to_indexed(masked_pattern):
    seq = []
    for ch in masked_pattern:
        if ch in ['_', '?']:
            seq.append(None)
        else:
            seq.append(char_to_int.get(ch))
    return seq

def compute_gamma_log(masked_idx_seq):
    T = len(masked_idx_seq)
    alpha = np.full((T, N), -np.inf)
    beta = np.full((T, N), -np.inf)

    # Forward
    if masked_idx_seq[0] is None:
        alpha[0] = log_start
    else:
        alpha[0] = log_start + log_emit[:, masked_idx_seq[0]]
    for t in range(1, T):
        o = masked_idx_seq[t]
        emit_log = 0.0 if o is None else log_emit[:, o]
        alpha[t] = emit_log + logsumexp(alpha[t - 1][:, None] + log_trans, axis=0)

    # Backward
    beta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        next_o = masked_idx_seq[t + 1]
        emit_next = 0.0 if next_o is None else log_emit[:, next_o]
        beta[t] = logsumexp(log_trans + (emit_next + beta[t + 1])[None, :], axis=1)

    gamma_log = alpha + beta
    gamma_log = np.clip(gamma_log, -1e10, 1e10)  # Prevent overflow
    gamma_log -= logsumexp(gamma_log, axis=1, keepdims=True)
    return gamma_log

def get_position_letter_marginals(masked_pattern):
    masked_idx = pattern_to_indexed(masked_pattern)
    gamma_log = compute_gamma_log(masked_idx)
    gamma = np.exp(gamma_log)
    gamma = np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)  # Handle NaN/inf

    marginals = []
    for t, obs in enumerate(masked_idx):
        if obs is None:
            probs = gamma[t] @ model.emissionprob_
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = np.clip(probs, 0.0, 1.0)  # Ensure valid probs
            probs /= probs.sum() + 1e-12
            marginals.append(probs)
        else:
            onehot = np.zeros(V)
            onehot[obs] = 1.0
            marginals.append(onehot)
    return marginals

def oracle(masked_word, guessed_letters):
    if '_' not in masked_word:
        return {}
    
    marginals = get_position_letter_marginals(masked_pattern=masked_word)
    blanks = [i for i, ch in enumerate(masked_word) if ch in ['_', '?']]
    if not blanks:
        probs_v = marginals[0]
    else:
        # Average emission probabilities across all blank positions for robustness
        probs_v = np.mean([marginals[pos] for pos in blanks], axis=0)

    probs_26 = np.array([probs_v[idx] for idx in letter_indexes])
    
    for i, ch in enumerate(list("abcdefghijklmnopqrstuvwxyz")):
        if ch in guessed_letters:
            probs_26[i] = 0.0
    
    total = probs_26.sum()
    if total == 0:
        probs_26 = np.ones(26) / 26.0
    else:
        probs_26 /= total + 1e-12
    
    probs_26 = np.nan_to_num(probs_26, nan=1.0/26.0, posinf=1.0/26.0, neginf=1.0/26.0)
    probs_26 = np.clip(probs_26, 0.0, 1.0)
    probs_26 /= probs_26.sum() + 1e-12  # Re-normalize
    
    return dict(zip(list("abcdefghijklmnopqrstuvwxyz"), probs_26))
