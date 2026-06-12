import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    x_norm = (x - mean) / np.sqrt(var + eps)

    return gamma * x_norm + beta
    

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # Linear projections
    Q_proj = np.matmul(Q, W_q)
    K_proj = np.matmul(K, W_k)
    V_proj = np.matmul(V, W_v)

    # Split heads
    Q_heads = Q_proj.reshape(batch_size, seq_len, num_heads, d_k)
    K_heads = K_proj.reshape(batch_size, seq_len, num_heads, d_k)
    V_heads = V_proj.reshape(batch_size, seq_len, num_heads, d_k)

    # (batch, heads, seq_len, d_k)
    Q_heads = Q_heads.transpose(0, 2, 1, 3)
    K_heads = K_heads.transpose(0, 2, 1, 3)
    V_heads = V_heads.transpose(0, 2, 1, 3)

    # Scaled dot-product attention
    scores = np.matmul(
        Q_heads,
        K_heads.transpose(0, 1, 3, 2)
    )

    scores = scores / np.sqrt(d_k)

    weights = softmax(scores, axis=-1)

    head_output = np.matmul(weights, V_heads)

    # Concatenate heads
    head_output = head_output.transpose(0, 2, 1, 3)

    concat = head_output.reshape(
        batch_size,
        seq_len,
        d_model
    )

    # Output projection
    return np.matmul(concat, W_o)
    

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    hidden = np.maximum(0, np.matmul(x, W1) + b1)
    return np.matmul(hidden, W2) + b2


def encoder_block(
    x,
    W_q, W_k, W_v, W_o,
    W1, b1, W2, b2,
    gamma1, beta1,
    gamma2, beta2,
    num_heads
):
    attn = multi_head_attention(
        x, x, x,
        W_q, W_k, W_v, W_o,
        num_heads
    )

    x1 = layer_norm(
        x + attn,
        gamma1,
        beta1
    )

    ffn = feed_forward(
        x1,
        W1, b1,
        W2, b2
    )

    out = layer_norm(
        x1 + ffn,
        gamma2,
        beta2
    )

    return out