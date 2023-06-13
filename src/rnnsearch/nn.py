"""RNNsearch model definition."""
import pytorch_lightning as pl

# import torch


class RNNSearch(pl.LightningModule):
    """RNNsearch model, as defined in the paper."""

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
