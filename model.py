import torch
import torch.nn as nn

class BearingFaultCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(BearingFaultCNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=16, padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, 1, window_size)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.global_avg_pool(x)      # shape: (batch_size, 64, 1)
        x = x.squeeze(-1)                 # shape: (batch_size, 64)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Quick sanity check: does the model run and produce the right output shape?
    model = BearingFaultCNN(num_classes=10)

    # Simulate a batch of 8 windows, each 1024 samples
    dummy_input = torch.randn(8, 1, 1024)
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # should be (8, 4)

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")