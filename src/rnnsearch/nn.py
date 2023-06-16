"""RNNsearch model definition."""
import functools
import itertools
import typing as tp

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnnutils


class Bridge(nn.Module):
    """Compute the intial state of the decoder given the last state of the encoder."""

    def __init__(self, input_size: int, output_size: int) -> None:
        """Intializes the weight matrix of the bridge.

        Args:
            input_size: The hidden size of the encoder.
            output_size: The dimensionality of the output. Should match the
                hidden size of the decoder.
        """
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)
        self.activation = nn.Tanh()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the bridge.

        Args:
            input: The last hidden state of the encoder.

        Returns:
            The first hidden state of the decoder.
        """
        # NOTE: on ignore[no-any-return], nn.Linear.forward is not typed
        # apparently.
        return self.activation(self.linear(input))  # type: ignore[no-any-return]


class _Reducer(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.v = nn.Parameter(torch.rand(input_size))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.matmul(input, self.v)


def _multiarange(counts: torch.Tensor) -> torch.Tensor:
    """Returns a sequence of aranges concatenated along the first dimension.

    This operation is vectorizes, so it doesn't actually concatenate aranges.

    >>> counts = torch.tensor([1, 3, 2])
    >>> _multiarange(counts)
    tensor([0, 0, 1, 2, 0, 1])

    >>> _multiarange(torch.tensor([3, 2, 1, 0]))
    tensor([0, 1, 2, 0, 1, 0])

    >>> _multiarange(torch.tensor([0]))
    tensor([], dtype=torch.int64)
    """
    counts = counts[torch.nonzero(counts, as_tuple=True)]
    if counts.nelement() == 0:
        return torch.tensor([], dtype=torch.int64)
    counts1 = counts[:-1]
    reset_index = counts1.cumsum(0)

    incr = torch.ones(int(counts.sum()), dtype=torch.int64)
    incr[0] = 0
    incr[reset_index] = 1 - counts1
    out: torch.Tensor = incr.cumsum(0)
    return out


class Alignment(nn.Module):
    """Alignment model, as described in the paper."""

    def __init__(self, encoder_size: int, decoder_size: int, hidden_size: int) -> None:
        """Initialize alignment weights.

        Args:
            encoder_size: The dimensionality of the hidden encoder states.
            decoder_size: The dimensionality of the hidden decoder states.
            hidden_size: The hidden size of the alignment model.
        """
        super().__init__()
        self.encoder_w = nn.Linear(encoder_size, hidden_size, bias=False)
        self.decoder_w = nn.Linear(decoder_size, hidden_size, bias=False)
        self.reducer = _Reducer(hidden_size)
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)

    @staticmethod
    def attention_mask(
        lens: torch.Tensor, max_len: tp.Union[int, torch.Tensor, None] = None
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention mask for a 2D tensor using the sequence lenghts.

        Assumes dimensions (L, B), where L is sequence length (padded) and B is
        batch size. ``lens`` should be a tensor of dimension (B, ), where each
        value represets the length of the sequence before padding.

        Args:
            lens: Tensor of shape (B,), represents the lenght of each sequence
                pre padding.
            max_len: if max_len is passed, defines dimension L instead of max(lens)

        Return:
            A tuple of tensors of shapes (cumsum() with padded positions as 1 and valid
            positions as 0.

        >>> mask = Alignment.attention_mask(torch.tensor([1, 3, 4]))
        >>> t = torch.zeros(4, 3)
        >>> t[mask] = 1.
        >>> t
        tensor([[0., 0., 0.],
                [1., 0., 0.],
                [1., 0., 0.],
                [1., 1., 0.]])
        >>> mask = Alignment.attention_mask(torch.tensor([1, 3, 4]), max_len=5)
        >>> t = torch.zeros(5, 3)
        >>> t[mask] = 1.
        >>> t
        tensor([[0., 0., 0.],
                [1., 0., 0.],
                [1., 0., 0.],
                [1., 1., 0.],
                [1., 1., 1.]])
        """
        max_len = max_len or torch.max(lens)
        lens_complement = max_len - lens

        offset = torch.repeat_interleave(lens, lens_complement)
        return (
            _multiarange(lens_complement) + offset,
            torch.repeat_interleave(lens_complement),
        )

    def forward(
        self,
        encoder_output: rnnutils.PackedSequence,
        last_decoder_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the alignment model.

        Computes the attention scores between the decoder hidden states and the
        encoder hidden states, and returns the weighted context vector.

        Args:
            encoder_output: All the hidden states of the encoder, h_j in the paper.
            last_decoder_hidden: The last hidden_state of the decoder, s_{i-1}
                in the paper.

        Return:
            The weighted context vector.
        """
        encoder_padded, encoder_lens = rnnutils.pad_packed_sequence(encoder_output)
        # NOTE: the paper mentions caching the tensors after linear layer for
        # the encoder hiddens. I'm not cool and efficient like them, so no
        # caching.
        energy_padded = self.reducer(
            self.activation(
                self.encoder_w(encoder_padded) + self.decoder_w(last_decoder_hidden)
            )
        )
        energy_padded[self.attention_mask(encoder_lens)] = -torch.inf
        scores_padded = self.softmax(energy_padded)

        context = torch.sum(scores_padded[..., None] * encoder_padded, dim=0)
        return context

# import torch


class RNNSearch(pl.LightningModule):
    """RNNSearch model, as defined in the paper."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        output_dim: int,
        alignment_dim: int,
    ) -> None:
        """Initialize an RNNSearch model."""
        super().__init__()
