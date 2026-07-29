#pepstl.py
import torch
import torch.nn as nn

class PepSTL(nn.Module):
    """
    PepMTL: Peptide Single Task Model.
    A single task neural network variant designed specifically for individual peptide
    bioactivity prediction. Utilizes the same hybrid backbone architecture (Bi-LSTM + 1D CNN)
    as PepMTL to ensure fair benchmark comparisons.
    Attributes:
        lstm: Bidirectional LSTM feature extractor.
        cnn: 1D convolutional feature mapping layer.
        global_pool: Global average pooling layer.
        norm: Layer normalization layer for pooled representations.
        head: Single task binary classification head.
    """
    def __init__(self,
                 input_dim: int = 1280,
                 lstm_num_layers: int = 2,
                 lstm_hidden: int = 512,
                 cnn_out: int = 256,
                 cnn_kernel: int = 3,
                 head_hidden: int = 512,
                 dropout: float = 0.3):
        """
        Initializes the PepSTL architecture with specified hyperparameter configurations.
        Args:
            input_dim: Dimension of input peptide sequence embeddings. Defaults to 1280.
            lstm_num_layers: Number of stacked LSTM layers. Defaults to 2.
            lstm_hidden: Hidden units per directional LSTM layer. Defaults to 512.
            cnn_out: Number of output channels in the 1D CNN layer. Defaults to 256.
            cnn_kernel: Kernel size for 1D convolution. Defaults to 3.
            head_hidden: Hidden unit dimensions within the output head. Defaults to 512.
            dropout: Dropout probability applied across layers. Defaults to 0.3.
        """

        super(PepSTL, self).__init__()

        # Shared Backbone
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0
        )

        self.cnn = nn.Sequential(
            nn.Conv1d(lstm_hidden * 2, cnn_out, kernel_size=cnn_kernel, padding=cnn_kernel // 2),
            nn.ReLU(),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(cnn_out)

        # Single binary output head
        self.head = nn.Sequential(
            nn.Linear(cnn_out, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)
        )

    # Perform forward pass through the backbone and single task binary prediction head.
    def forward(self, x):
        # Automatically expand 2D mean-pooled embeddings into 3D pseudo-sequence tensor
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        cnn_out = self.cnn(lstm_out.transpose(1, 2))
        pooled = self.global_pool(cnn_out).squeeze(-1)
        out = self.head(pooled)

        # Returns (batch_size,) tensor and None for pipeline compatibility
        return out.squeeze(-1), None