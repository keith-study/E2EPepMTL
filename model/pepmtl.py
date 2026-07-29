#pepmtl.py
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class PepMTL(nn.Module):
    """
    PepMTL: Peptide Multi-Task Model.
    A shared-backbone multi-task neural network designed for peptide bioactivity
    prediction. Features a hybrid feature extraction backbone (Bi-LSTM + 1D-CNN),
    gradient checkpointing for GPU memory efficiency, and parallel task prediction heads.
    Attributes:
        num_tasks : Number of target bioactivity prediction tasks.
        shared_lstm: Shared bidirectional LSTM feature extractor.
        shared_cnn: Shared 1D convolutional feature mapping layer.
        global_pool: Global average pooling layer.
        norm: Layer normalization layer for pooled feature embeddings.
        task_head: Primary multi-task classification head.
        mlp_head: Auxiliary multi-task classification head for regularization.
    """

    def __init__(self,
                 input_dim: int = 1280,
                 num_tasks: int = 16,
                 lstm_num_layers: int = 2,
                 lstm_hidden: int = 512,
                 cnn_out: int = 256,
                 cnn_kernel: int = 3,
                 head_hidden: int = 512,
                 dropout: float = 0.3):
        """
        Initializes the PepMTL architecture with specified hyperparameter configurations.
        Args:
            input_dim (int, optional): Dimension of input peptide sequence embeddings. Defaults to 1280.
            num_tasks (int, optional): Total number of output classification tasks. Defaults to 16.
            lstm_num_layers (int, optional): Number of stacked LSTM layers. Defaults to 2.
            lstm_hidden (int, optional): Hidden units per directional LSTM layer. Defaults to 512.
            cnn_out (int, optional): Number of output channels in the 1D CNN layer. Defaults to 256.
            cnn_kernel (int, optional): Kernel size for 1D convolution. Defaults to 3.
            head_hidden (int, optional): Hidden unit dimensions within prediction heads. Defaults to 512.
            dropout (float, optional): Dropout probability applied across layers. Defaults to 0.3.
        """

        super(PepMTL, self).__init__()

        self.num_tasks = num_tasks

        # Shared Backbone
        self.shared_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0
        )

        self.shared_cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=lstm_hidden * 2,
                out_channels=cnn_out,
                kernel_size=cnn_kernel,
                padding=cnn_kernel // 2
            ),
            nn.ReLU(),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(cnn_out)

        # Primary Multi-Task Head
        self.task_head = nn.Sequential(
            nn.Linear(cnn_out, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_tasks)
        )

        # Auxiliary Multi-Task Head
        self.mlp_head = nn.Sequential(
            nn.Linear(cnn_out, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden // 2, num_tasks)
        )

    # Extract pooled backbone features from sequence input tensors.
    def _forward_backbone(self, x):
        lstm_out, _ = self.shared_lstm(x)
        cnn_out = self.shared_cnn(lstm_out.transpose(1, 2))
        pooled = self.global_pool(cnn_out).squeeze(-1)
        return pooled

    # Perform forward pass through the shared backbone and multi-task prediction heads.
    def forward(self, x):
        # Automatically expand 2D mean-pooled embeddings into 3D pseudo-sequence tensor
        if x.dim() == 2:
            x = x.unsqueeze(1)   # (batch, 1, 1280)

        # Execute shared backbone with activation checkpointing during training
        if self.training:
            pooled = checkpoint(self._forward_backbone, x, use_reentrant=False)
        else:
            pooled = self._forward_backbone(x)

        task_pred = self.task_head(pooled)
        mlp_pred = self.mlp_head(pooled)

        return task_pred, mlp_pred