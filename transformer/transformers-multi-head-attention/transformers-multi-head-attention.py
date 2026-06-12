import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    Q_proj = np.matmul(Q, W_q)
    K_proj = np.matmul(K, W_k)
    V_proj = np.matmul(V, W_v)

    Q_heads = Q_proj.reshape(batch_size, seq_len, num_heads, d_k)
    K_heads = K_proj.reshape(batch_size, seq_len, num_heads, d_k)
    V_heads = V_proj.reshape(batch_size, seq_len, num_heads, d_k)

    Q_heads = Q_heads.transpose(0, 2, 1, 3)
    K_heads = K_heads.transpose(0, 2, 1, 3)
    V_heads = V_heads.transpose(0,2, 1, 3)

    scores = np.matmul(Q_heads, K_heads.transpose(0, 1, 3, 2))
    scores = scores / np.sqrt(d_k)

    attention_weights = softmax(scores, axis=-1)

    head_outputs = np.matmul(attention_weights, V_heads)

    # Concatenate heads
    head_outputs = head_outputs.transpose(0, 2, 1, 3)
    concat = head_outputs.reshape(batch_size, seq_len, d_model)

    # Final output projection
    output = np.matmul(concat, W_o)

    return output
    