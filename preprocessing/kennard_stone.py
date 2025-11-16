import numpy as np


def kennard_stone(X, k):
    n = X.shape[0]
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    i0, i1 = np.unravel_index(np.argmax(D), D.shape)
    selected = [i0, i1]
    remaining = list(set(range(n)) - set(selected))
    while len(selected) < k and remaining:
        dmin = np.min(D[remaining][:, selected], axis=1)
        next_idx = remaining[int(np.argmax(dmin))]
        selected.append(next_idx)
        remaining.remove(next_idx)
    return np.array(selected), np.array(remaining)
