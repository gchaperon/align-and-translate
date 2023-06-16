import pytest
import torch
import torch.nn.utils.rnn as rnnutils
from hypothesis import given, settings
from hypothesis import strategies as st

import rnnsearch.nn as nn


@settings(max_examples=20)
@given(
    st.integers(min_value=1, max_value=20),
    st.integers(min_value=1, max_value=20),
    st.integers(min_value=1, max_value=8),
    st.booleans(),
)
@torch.no_grad()
def test_bridge(
    input_size: int, output_size: int, batch_size: int, batched: bool
) -> None:
    bridge = nn.Bridge(input_size, output_size)

    input = torch.rand(batch_size, input_size) if batched else torch.rand(input_size)
    output = bridge(input)

    assert output.shape == (batch_size, output_size) if batched else (output_size,)
    assert torch.all(output >= -1) and torch.all(output <= 1)


@settings(max_examples=20)
@given(
    st.data(),
    st.integers(1, 20),
    st.integers(1, 20),
    st.integers(1, 20),
    st.integers(1, 8),
)
@torch.no_grad()
def test_alignment(
    data: st.DataObject,
    encoder_size: int,
    decoder_size: int,
    hidden_size: int,
    batch_size: int,
) -> None:

    box: dict[str, torch.Tensor] = {}
    model = nn.Alignment(encoder_size, decoder_size, hidden_size)
    model.softmax.register_forward_hook(
        lambda _, __, output: box.setdefault("value", output)
    )
    encoder_output = rnnutils.pack_sequence(
        [
            torch.rand(length, encoder_size)
            for length in data.draw(
                st.lists(st.integers(1, 20), min_size=batch_size, max_size=batch_size)
            )
        ],
        enforce_sorted=False,
    )
    last_decoder_hidden = torch.rand(batch_size, decoder_size)
    context = model(encoder_output, last_decoder_hidden)
    scores = box["value"]

    assert isinstance(context, torch.Tensor)

    assert context.shape == (batch_size, encoder_size)
    # score should add up to one
    assert torch.allclose(torch.sum(scores, dim=0), torch.tensor(1.0))


@settings(max_examples=20)
@given(*(st.integers(min_value=1, max_value=20) for _ in range(5)))
@torch.no_grad()
def test_rnnsearch_model_init(
    vocab_size: int,
    embedding_dim: int,
    hidden_size: int,
    output_dim: int,
    alignment_dim: int,
) -> None:
    nn.RNNSearch(vocab_size, embedding_dim, hidden_size, output_dim, alignment_dim)


@st.composite
def sentence_batches(draw: st.DrawFn) -> tuple[list[list[int]], list[list[int]], int]:
    vocab_size = draw(st.integers(1, 100))
    # NOTE: RNNs don't handle 0-length inputs
    sentences = st.lists(st.integers(0, vocab_size - 1), min_size=1, max_size=100)
    batch = draw(st.lists(sentences, min_size=1, max_size=8))
    batch_size = len(batch)
    target = [draw(sentences) for _ in range(batch_size)]
    return batch, target, vocab_size


@pytest.mark.xfail()
@settings(max_examples=20)
@given(sentence_batches(), *(st.integers(min_value=1, max_value=20) for _ in range(4)))
@torch.no_grad()
def test_rnnsearch_forward(
    batch_and_vsize: tuple[list[list[int]], list[list[int]], int],
    embedding_dim: int,
    hidden_size: int,
    output_dim: int,
    alignment_dim: int,
) -> None:
    input_batch, decoder_batch, vocab_size = batch_and_vsize
    model = nn.RNNSearch(
        vocab_size, embedding_dim, hidden_size, output_dim, alignment_dim
    )
    input = rnnutils.pack_sequence(
        [torch.tensor(s) for s in input_batch], enforce_sorted=False
    )
    decoder_input = rnnutils.pack_sequence(
        [torch.tensor(s) for s in decoder_batch], enforce_sorted=False
    )

    output = model(input, decoder_input)
    assert isinstance(output, rnnutils.PackedSequence)
    assert len(input_batch) == output.batch_sizes[0]
