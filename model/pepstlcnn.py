#pepstlcnn.py
import torch
import torch.nn as nn
class PepSTL_CNN(nn.Module):
    """
    PepSTL_CNN: 1D CNN Single Task Model.
    A single task model architecture that relies strictly on stacked 1D convolutional
    layers, global average pooling, and layer normalization. Eliminates Bi-LSTM recurrent
    layers to evaluate the isolated performance of pure convolutional feature extraction
    on ESM-2 sequence embeddings.
    Attributes:
        cnn: Stacked 1D convolutional feature extraction layers.
        global_pool: Global average pooling layer.
        norm: Layer normalization layer for pooled feature maps.
        head: Single-task binary classification head.
    """
    def __init__(self,
                 input_dim: int = 1280,
                 cnn_out: int = 256,
                 cnn_kernel: int = 3,
                 head_hidden: int = 512,
                 dropout: float = 0.3):
        """
        Initializes the PepSTL_CNN architecture with specified hyperparameter configurations.
        Args:
            input_dim: Dimension of input peptide sequence embeddings. Defaults to 1280.
            cnn_out: Number of output channels in the 1D CNN layers. Defaults to 256.
            cnn_kernel: Kernel size for 1D convolutions. Defaults to 3.
            head_hidden: Hidden unit dimensions within the output head. Defaults to 512.
            dropout: Dropout probability applied across layers. Defaults to 0.3.
        """

        super(PepSTL_CNN, self).__init__()

        # Shared Backbone
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_out, kernel_size=cnn_kernel, padding=cnn_kernel // 2),
            nn.ReLU(),
            nn.Conv1d(cnn_out, cnn_out, kernel_size=cnn_kernel, padding=cnn_kernel // 2),  # 可加一層增強
            nn.ReLU(),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(cnn_out)

        # Single Binary Output Head
        self.head = nn.Sequential(
            nn.Linear(cnn_out, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)
        )

    # Perform forward pass through the 1D CNN backbone and single-task prediction head.
    def forward(self, x):
        # Automatically expand 2D mean-pooled embeddings into 3D pseudo-sequence tensor
        if x.dim() == 2:
            x = x.unsqueeze(1)
        cnn_out = self.cnn(x.transpose(1, 2))
        pooled = self.global_pool(cnn_out).squeeze(-1)
        pooled = self.norm(pooled)
        out = self.head(pooled)

        return out.squeeze(-1), None