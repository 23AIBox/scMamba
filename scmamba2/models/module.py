import torch
import torch.nn as nn


class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        hidden_dim: int = 512,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            if i == 0:
                self._decoder.append(nn.Linear(d_model, hidden_dim))
            else:
                self._decoder.append(nn.Linear(hidden_dim, hidden_dim))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(hidden_dim))
            # self._decoder.append(nn.Linear(d_model, d_model))
        self.out_layer = nn.Linear(hidden_dim, n_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)