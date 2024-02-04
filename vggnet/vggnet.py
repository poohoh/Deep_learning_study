import torch
import torch.nn as nn

VGG_Types = {
    'VGG16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

class VGG_16(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, init_weights=True):
        super(VGG_16, self).__init__()
        self.in_channels = in_channels

        # layer 생성
        self.conv_layers = self.create_layers(VGG_Types['VGG16'])

        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    #
    #     # weight initialize
    #     if init_weights:
    #         self._initialize_weights()
    #
    # def _initialize_weights(self):
    #


    def create_layers(self, architecture=VGG_Types['VGG16']):
        layers = []
        in_channels = self.in_channels  # 3

        for x in architecture:
            if type(x) == int:
                out_channels = x  # 채널 수는 유지됨

                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]

        return nn.Sequential(*layers)