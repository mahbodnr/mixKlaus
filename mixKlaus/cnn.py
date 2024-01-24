import torch
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self, layers, dropout=0.0, batchnorm=False, activation=nn.ReLU()):
        super(ANN, self).__init__()
        self.blocks = nn.ModuleList()
        for feature_idx in range(len(layers) - 1):
            self.blocks.append(nn.Linear(layers[feature_idx], layers[feature_idx + 1]))
            if batchnorm:
                self.blocks.append(nn.BatchNorm1d(layers[feature_idx + 1]))
            self.blocks.append(activation)
            if dropout:
                self.blocks.append(nn.Dropout(dropout))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    
    
class CNN(nn.Module):
    def __init__(
        self,
        features,
        kernel_size=3,
        batchnorm=True,
        activation=nn.ReLU(),
        pooling=True,
    ):
        super(CNN, self).__init__()
        self.features = features
        self.blocks = nn.ModuleList()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * (len(features) - 1)
        assert len(kernel_size) == len(features) - 1
        for feature_idx in range(len(features) - 1):
            self.blocks.append(
                nn.Conv2d(
                    features[feature_idx],
                    features[feature_idx + 1],
                    kernel_size=kernel_size[feature_idx],
                )
            )
            if batchnorm:
                self.blocks.append(nn.BatchNorm2d(features[feature_idx + 1]))
            self.blocks.append(activation)
            if pooling:
                self.blocks.append(nn.MaxPool2d(2, 2))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class BaselineCNN(nn.Module):
    def __init__(
        self,
        input_shape,
        cnn_features=[64, 128, 256],
        ann_layers=[1024, 256, 64, 10],
    ):
        super(BaselineCNN, self).__init__()
        self.conv = CNN([input_shape[0]] + cnn_features)
        self.ann = ANN(
            [
                self.calculate_last_layer_size(
                    input_shape,
                    cnn_features,
                )
            ]
            + ann_layers
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.ann(x)
        return x

    @staticmethod
    def calculate_last_layer_size(input_shape, cnn_features):
        x = torch.randn(1, *input_shape)
        x = CNN([input_shape[0]] + cnn_features)(x)
        return x.reshape(x.shape[0], -1).shape[1]
