import numpy as np
import joblib
from scipy.special import logsumexp

# Load trained model + mappings
obj = joblib.load("models/hmm_model.pkl")
model = obj["model"]
char_to_int = obj["char_to_int"]
int_to_char = obj["int_to_char"]

# Parameters
N = model.n_components
V = model.emissionprob_.shape[1]
log_start = np.log(model.startprob_ + 1e-12)
log_trans = np.log(model.transmat_ + 1e-12)
log_emit = np.log(model.emissionprob_ + 1e-12)

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
    beta  = np.full((T, N), -np.inf)
    # forward
    if masked_idx_seq[0] is None:
        alpha[0] = log_start
    else:
        alpha[0] = log_start + log_emit[:, masked_idx_seq[0]]
    for t in range(1, T):
        o = masked_idx_seq[t]
        emit_log = 0.0 if o is None else log_emit[:, o]
        alpha[t] = emit_log + logsumexp(alpha[t-1][:, None] + log_trans, axis=0)
    # backward
    beta[T-1] = 0.0
    for t in range(T-2, -1, -1):
        next_o = masked_idx_seq[t+1]
        emit_next = 0.0 if next_o is None else log_emit[:, next_o]
        beta[t] = logsumexp(log_trans + (emit_next + beta[t+1])[None, :], axis=1)
    gamma_log = alpha + beta
    gamma_log -= logsumexp(gamma_log, axis=1, keepdims=True)
    return gamma_log

def get_position_letter_marginals(masked_pattern):
    masked_idx = pattern_to_indexed(masked_pattern)
    gamma_log = compute_gamma_log(masked_idx)
    gamma = np.exp(gamma_log)
    marginals = []
    for t, obs in enumerate(masked_idx):
        if obs is None:
            probs = gamma[t] @ model.emissionprob_
            probs /= probs.sum() + 1e-12
            marginals.append(probs)
        else:
            onehot = np.zeros(V)
            onehot[obs] = 1.0
            marginals.append(onehot)
    return marginals

def oracle(masked_word, guessed_letters):
    """Return {letter: probability} for the next guesses."""
    marginals = get_position_letter_marginals(masked_word)
    blanks = [i for i, ch in enumerate(masked_word) if ch in ['_', '?']]
    if not blanks:
        return {}
    pos = blanks[0]
    probs_v = marginals[pos]
    probs_26 = np.array([probs_v[idx] for idx in letter_indexes])
    for i, ch in enumerate(list("abcdefghijklmnopqrstuvwxyz")):
        if ch in guessed_letters:
            probs_26[i] = 0.0
    probs_26 /= probs_26.sum() + 1e-12
    letters = list("abcdefghijklmnopqrstuvwxyz")
    return dict(sorted(zip(letters, probs_26), key=lambda x: x[1], reverse=True))

# Optional demo
if __name__ == "__main__":
    print(oracle("_a_e", {"a", "e"}))
