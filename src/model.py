import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, init_scale=1.0, beta=0.1):
        super(SimpleCNN, self).__init__()

        self.beta = beta

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(64 * 28 * 28, 1)

        self._initialize_weights(init_scale)

        self.last_conv3_activation = None

    def _initialize_weights(self, scale):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                m.weight.data *= scale
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def activation(self, x):
        return F.leaky_relu(x, negative_slope=self.beta)

    def forward(self, x):

        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))

        x = self.activation(self.conv3(x))

        # Store conv3 activation for metrics
        self.last_conv3_activation = x

        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def get_last_conv3_activation(self):
        return self.last_conv3_activation