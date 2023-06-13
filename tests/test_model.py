from hypothesis import given
from hypothesis import strategies as st

from rnnsearch.nn import RNNSearch


@given(*(st.integers(min_value=1, max_value=20) for _ in range(5)))
def test_rnnsearch_model_init(
    vocab_size: int,
    embedding_dim: int,
    hidden_size: int,
    output_dim: int,
    alignment_dim: int,
) -> None:
    RNNSearch(vocab_size, embedding_dim, hidden_size, output_dim, alignment_dim)
