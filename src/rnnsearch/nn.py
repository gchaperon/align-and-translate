"""RNNsearch model definition."""
import functools
import operator
import typing as tp

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnnutils

import rnnsearch.datasets as datasets


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

    def extra_repr(self) -> str:
        return f"input_size={self.v.shape[0]}"


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

    @functools.lru_cache(maxsize=1)
    def _encoder_pad_and_project(
        self, encoder_output: rnnutils.PackedSequence
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the encoder weight and cache it.

        Given maxsize=1, this essentially caches the application of the
        encoder weight in the sequential decoding steps.
        """
        encoder_padded, encoder_lens = rnnutils.pad_packed_sequence(encoder_output)
        projection: torch.Tensor = self.encoder_w(encoder_padded)
        return encoder_padded, encoder_lens, projection

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
        (
            encoder_padded,
            encoder_lens,
            encoder_projection,
        ) = self._encoder_pad_and_project(encoder_output)
        energy_padded = self.reducer(
            self.activation(encoder_projection + self.decoder_w(last_decoder_hidden))
        )
        energy_padded[self.attention_mask(encoder_lens)] = -torch.inf
        scores_padded = self.softmax(energy_padded)

        context = torch.sum(scores_padded[..., None] * encoder_padded, dim=0)
        return context


class AttentionDecoder(nn.Module):
    """Decoder using the attention mechanism.

    At each step of the decoding process, the model computes an alignment
    scores w/r to the encoder  hidden states, and uses them to compute a new
    context vector.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        encoder_hidden_size: int,
        alignment_dim: int,
    ) -> None:
        """Create a new AttentionDecoder module.

        Args:
            input_size: The embedding dimension of each vector in the input.
                E*y_i in section A.2.2 of the paper.
            hidden_size: The hidden size of the network. Dimension of the s_i vectors.
            encoder_hidden_size: The encoder hidden size. Dimesion of the h_j vectors.
            alignment_dim: The hidden size of the alignment model. Dimension n'
                in the alignment model.
        """
        super().__init__()
        self.alignment = Alignment(
            encoder_size=encoder_hidden_size,
            decoder_size=hidden_size,
            hidden_size=alignment_dim,
        )
        self.decoder_cell = nn.GRUCell(
            input_size=input_size + encoder_hidden_size, hidden_size=hidden_size
        )
        self.activation = nn.Tanh()

    def forward(
        self,
        encoder_output: rnnutils.PackedSequence,
        decoder_input: rnnutils.PackedSequence,
        h0: torch.Tensor,
    ) -> rnnutils.PackedSequence:
        """Forward pass of the attention decoder.

        Uses and alignment model at each decoding step to compute a dynamic
        context vector.

        Args:
            encoder_output: The hidden states of the encoder.
            decoder_input: The target sequence for the decoder. For each word
                in the decoder input, this model produces a hidden state, later
                used to predict the next word. Essentially this "shifts" the
                input one position.
            h0: The initial hidden state of the network.

        Return:
            The encoder hidden states, from h_1 to h_n, for each sequence in
            the batch. The value of n depends on the length of each sequence in
            the batch.
        """
        input_padded, decoder_lens = rnnutils.pad_packed_sequence(decoder_input)

        hiddens: list[torch.Tensor] = []
        hi = h0
        for word in input_padded.unbind(0):
            context = self.alignment(encoder_output, hi)
            hi = self.decoder_cell(torch.concat([word, context], dim=1), hi)
            hiddens.append(hi)

        return rnnutils.pack_padded_sequence(
            torch.stack(hiddens), decoder_lens, enforce_sorted=False
        )


def _make_classifier(input_features: int, output_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_features, 500),
        nn.ReLU(),
        nn.Linear(500, output_features),
    )


class RNNSearch(pl.LightningModule):
    """RNNSearch model, as defined in the paper."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        output_dim: int,
        alignment_dim: int,
        *,
        learn_rate: float,
    ) -> None:
        """Initialize an RNNSearch model."""
        super().__init__()
        self.save_hyperparameters()
        # NOTE: shared embedding matrix for source and target language
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.encoder = nn.GRU(
            input_size=embedding_dim, hidden_size=hidden_size, bidirectional=True
        )
        self.bridge = Bridge(input_size=hidden_size, output_size=hidden_size)
        self.decoder = AttentionDecoder(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            encoder_hidden_size=2 * hidden_size,
            alignment_dim=alignment_dim,
        )
        self.classifier = _make_classifier(hidden_size, output_dim)
        self.loss = nn.CrossEntropyLoss()

        self.learn_rate = learn_rate

    def forward(
        self,
        encoder_input: rnnutils.PackedSequence,
        decoder_input: rnnutils.PackedSequence,
    ) -> rnnutils.PackedSequence:
        """Forward pass for the RNNSeach architecture.

        The input sequence is encoded using a bidirectional GRU network.
        Teacher forcing is then used for the decoding process, where the whole
        target sequence is passed to the decoder ant its job is to "shift" it
        by one position.

        The last hidden unit in the backward direction is passed through a
        bridge network and used as the initial hidden state for the decoder.

        The hidden states of the decoder are passed through a simple linear
        clssifier to get the final logits.

        Args:
            encoder_input: The input sequence.
            decoder_input: Input for the decoder. Used for teacher forcing.

        Return:
            The decoded logits.
        """
        encoder_output, last_hidden = self.encoder(
            encoder_input._replace(data=self.embedding(encoder_input.data))
        )
        decoder_output = self.decoder(
            encoder_output,
            decoder_input._replace(data=self.embedding(decoder_input.data)),
            self.bridge(last_hidden[1]),
        )
        logits: rnnutils.PackedSequence = decoder_output._replace(
            data=self.classifier(decoder_output.data)
        )
        return logits

    def training_step(self, batch: datasets.TrainItem, _: int) -> torch.Tensor:
        """Train step."""
        input, teacher_forcing, target = operator.attrgetter("fr", "en", "target")(
            batch
        )
        logits = self(input, teacher_forcing)
        loss: torch.Tensor = self.loss(logits.data, target.data)
        self.log("train/loss", loss, batch_size=target.batch_sizes[0], prog_bar=True)
        return loss

    def validation_step(self, batch: datasets.TrainItem, _: int) -> None:
        """Validation step."""
        input, teacher_forcing, target = operator.attrgetter("fr", "en", "target")(
            batch
        )
        logits = self(input, teacher_forcing)
        loss: torch.Tensor = self.loss(logits.data, target.data)
        self.log("val/loss", loss, batch_size=target.batch_sizes[0])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Uptimizer for the network."""
        return torch.optim.Adam(self.parameters(), lr=self.learn_rate)
