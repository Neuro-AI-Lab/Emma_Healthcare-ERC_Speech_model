import torch
import torch.nn as nn

class EmotionCNN(nn.Module):
    def __init__(self, input_dim, num_classes=7):
        super(EmotionCNN, self).__init__()

        # Conv 블록들을 입력 길이에 상관없이 동작하도록 정의
        self.layer1 = self._block(1, 512, 5)
        self.layer2 = self._block(512, 256, 5, dropout=0.3)
        self.layer3 = self._block(256, 128, 3, dropout=0.3)
        self.layer4 = self._block(128, 64, 3, dropout=0.3)

        # 4번의 풀링을 거친 후의 차원을 자동으로 계산
        self.flatten_dim = self._get_flatten_size(input_dim)

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def _block(self, in_c, out_c, k, dropout=None):
        layers = [
            nn.Conv1d(in_c, out_c, kernel_size=k, padding=k//2),
            nn.BatchNorm1d(out_c),
            nn.ReLU(),
            nn.MaxPool1d(2)
        ]
        if dropout: layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def _get_flatten_size(self, input_dim):
        with torch.no_grad():
            x = torch.zeros(1, 1, input_dim)
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            return x.numel()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, 1)
        return self.fc(x)