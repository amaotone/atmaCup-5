import torch
from torch import nn


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, dilation=1, dropout=0.0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, filter_size, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.model(x)


class Model(nn.Module):
    def __init__(
        self, in_channels, n_cont_features, kernel_sizes, hidden_channels=64, out_dim=1
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        for kernel_size in self.kernel_sizes:
            setattr(
                self,
                f"seq{kernel_size}",
                nn.Sequential(
                    Conv1dBlock(in_channels, hidden_channels, kernel_size, dropout=0.2),
                    Conv1dBlock(
                        hidden_channels, hidden_channels, kernel_size, dropout=0.2
                    ),
                ),
            )
        self.cont = nn.Sequential(
            nn.Linear(n_cont_features, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.last_linear = nn.Sequential(
            nn.Linear(hidden_channels * (len(self.kernel_sizes) + 1), hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_dim),
            nn.Sigmoid(),
        )

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x_seq, x_cont):
        outs = []
        for filter_size in self.kernel_sizes:
            out = getattr(self, f"seq{filter_size}")(x_seq)
            out, _ = torch.max(out, -1)
            outs.append(out)

        outs.append(self.cont(x_cont))
        out = torch.cat(outs, axis=1)
        out = self.last_linear(out)
        return out.flatten()
