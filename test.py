import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        out = self.dropout(out)
        return out

class TCN(nn.Module):  
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn_layers = self._create_tcn(input_size, output_size, num_channels, kernel_size, dropout)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def _create_tcn(self, input_size, output_size, num_channels, kernel_size, dropout):
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [ResidualBlock(in_channels, out_channels, kernel_size, dilation_size)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.tcn_layers(x)
        out = out[:, :, -1]  # Take the last timestep
        out = self.fc(out)
        return out

# Example usage
input_size = 10  # Input size (number of features)
output_size = 5  # Output size (number of classes)
num_channels = [64, 64, 64]  # Number of channels in each residual block
kernel_size = 3  # Kernel size for convolutional layers
dropout = 0.2  # Dropout rate

model = TCN(input_size, output_size, num_channels, kernel_size, dropout)
input_tensor = torch.randn(16, input_size, 100)  # Batch size x Input size x Sequence length
output = model(input_tensor)
print(output.shape)  # Output shape: Batch size x Output size
