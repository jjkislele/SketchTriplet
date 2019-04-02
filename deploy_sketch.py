import torch.nn as nn

class SketchTriplet(nn.Module):
    def __init__(self, num_feat):
        super(SketchTriplet, self).__init__()
        self.num_feat = num_feat
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=15,
                stride=3,
                padding=0
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2
            )
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=0
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2
            )
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2
            )
        )
        self.fc6 = nn.Sequential(
            nn.Linear(256 * 4 * 4, 2048),
            nn.ReLU(inplace=True)
        )
        self.fc7 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True)
        )
        self.feat = nn.Sequential(
            nn.Linear(512, num_feat),
            nn.ReLU(inplace=True)
        )

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.fc6(x)
            x = self.fc7(x)
            x = self.feat(x)
            return x
