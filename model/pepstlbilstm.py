#pepstlbilstm.py
import torch
import torch.nn as nn
class PepSTL_BiLSTM(nn.Module):
    """
    PepSTL_BiLSTM: Bi-LSTM Single Task Model.
    A single task model architecture that relies strictly on a Bi-LSTM
    backbone and sequence-level mean pooling, eliminating convolutional (CNN) layers to evaluate
    the isolated contribution of recurrent feature processing.
    Attributes:
        lstm: Bidirectional LSTM sequence feature extractor.
        head: Single task binary classification head.
    """
    def __init__(self,
                 input_dim: int = 1280,
                 lstm_num_layers: int = 2,
                 lstm_hidden: int = 512,
                 head_hidden: int = 512,
                 dropout: float = 0.3):
        """
        Initializes the PepSTL_BiLSTM architecture with specified hyperparameter configurations.
        Args:
            input_dim: Dimension of input peptide sequence embeddings. Defaults to 1280.
            lstm_num_layers: Number of stacked LSTM layers. Defaults to 2.
            lstm_hidden: Hidden units per directional LSTM layer. Defaults to 512.
            head_hidden: Hidden unit dimensions within the output head. Defaults to 512.
            dropout: Dropout probability applied across layers. Defaults to 0.3.
        """

        super(PepSTL_BiLSTM, self).__init__()

        # Shared Backbone
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0
        )

        # Single binary output head
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)
        )

    # Perform forward pass through the Bi-LSTM backbone and single task prediction head.
    def forward(self, x):
        # Automatically expand 2D mean-pooled embeddings into 3D pseudo-sequence tensor
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)    # (batch, seq_len, 2*lstm_hidden)
        pooled = torch.mean(lstm_out, dim=1)    # mean pooling 取代 CNN + global pool
        out = self.head(pooled)

        return out.squeeze(-1), None