import datetime as dt
import typing as tp

import torch
import torch.nn.utils.rnn as rnnutils
from hypothesis import given, settings
from hypothesis import strategies as st

import rnnsearch.nn as nn

default_settings = settings(max_examples=20, deadline=dt.timedelta(seconds=1))


@default_settings
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


def sequence_batch(vector_size: int, lengths: tp.List[int]) -> rnnutils.PackedSequence:
    return rnnutils.pack_sequence(
        [torch.rand(length, vector_size) for length in lengths],
        enforce_sorted=False,
    )


@default_settings
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
    encoder_output = sequence_batch(
        vector_size=encoder_size,
        lengths=data.draw(
            st.lists(st.integers(1, 20), min_size=batch_size, max_size=batch_size)
        ),
    )
    last_decoder_hidden = torch.rand(batch_size, decoder_size)
    context = model(encoder_output, last_decoder_hidden)
    scores = box["value"]

    assert isinstance(context, torch.Tensor)

    assert context.shape == (batch_size, encoder_size)
    # score should add up to one
    assert torch.allclose(torch.sum(scores, dim=0), torch.tensor(1.0))


@default_settings
@given(
    st.data(),
    st.integers(1, 20),
    st.integers(1, 20),
    st.integers(1, 20),
    st.integers(1, 20),
    st.integers(1, 8),
)
@torch.no_grad()
def test_attention_decoder(
    data: st.DataObject,
    input_size: int,
    hidden_size: int,
    encoder_hidden_size: int,
    alignment_dim: int,
    batch_size: int,
) -> None:
    model = nn.AttentionDecoder(
        input_size, hidden_size, encoder_hidden_size, alignment_dim
    )
    lengths = st.lists(st.integers(1, 20), min_size=batch_size, max_size=batch_size)
    encoder_output = sequence_batch(
        vector_size=encoder_hidden_size, lengths=data.draw(lengths)
    )
    decoder_input = sequence_batch(vector_size=input_size, lengths=data.draw(lengths))
    h0 = torch.rand(batch_size, hidden_size)

    output = model(encoder_output, decoder_input, h0)

    assert isinstance(output, rnnutils.PackedSequence)
    assert output.batch_sizes[0] == batch_size

    output_padded, output_lens = rnnutils.pad_packed_sequence(output)
    input_padded, input_lens = rnnutils.pad_packed_sequence(decoder_input)

    assert (
        output_padded.shape[:2] == input_padded.shape[:2]
    ), "max sequence length and batch size should batch"
    assert torch.all(output_lens == input_lens)


@default_settings
@given(
    st.data(),
    st.integers(min_value=2, max_value=20),  # There is no support for vocab size of 1
    *(st.integers(min_value=1, max_value=20) for _ in range(4)),
    st.integers(1, 8),
)
@torch.no_grad()
def test_rnnsearch_model(
    data: st.DataObject,
    vocab_size: int,
    embedding_dim: int,
    hidden_size: int,
    output_dim: int,
    alignment_dim: int,
    batch_size: int,
) -> None:
    model = nn.RNNSearch(
        vocab_size,
        embedding_dim,
        hidden_size,
        output_dim,
        alignment_dim,
        learn_rate=1e-3,
    )
    lengths = st.lists(st.integers(1, 20), min_size=batch_size, max_size=batch_size)
    input = rnnutils.pack_sequence(
        [
            torch.randint(low=0, high=vocab_size, size=(length,))
            for length in data.draw(lengths)
        ],
        enforce_sorted=False,
    )
    teacher_forcing = rnnutils.pack_sequence(
        [
            torch.randint(low=0, high=vocab_size, size=(length,))
            for length in data.draw(lengths)
        ],
        enforce_sorted=False,
    )
    output = model(input, teacher_forcing)

    assert isinstance(output, rnnutils.PackedSequence)

    output_padded, lens = rnnutils.pad_packed_sequence(output)
    assert output_padded.shape[1] == batch_size
    assert output_padded.shape[2] == output_dim
    assert torch.all(lens == rnnutils.pad_packed_sequence(teacher_forcing)[1])
