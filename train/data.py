import numpy as np


def draw_random_batch(
    tokens: np.ndarray,
    batch_size: int,
    seq_len: int,
) -> np.ndarray:
    """Draw batch_size random contiguous windows of seq_len tokens.

    Returns a (batch, seq) int64 NumPy array on the host. tokens is the
    uint16 stream from scripts/prepare_data.py, e.g.
    np.memmap("train.bin", dtype=np.uint16, mode="r"). Windows span
    document boundaries (the reference's "wrap" packing).
    """
    starts = np.random.randint(0, len(tokens) - seq_len + 1, size=batch_size)
    batch = np.stack([tokens[s:s + seq_len] for s in starts]).astype(np.int64)

    return batch
