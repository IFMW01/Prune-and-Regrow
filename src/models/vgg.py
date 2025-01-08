import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, vgg_name, vgg_dict, num_classes, dropout=0.0):
        super(VGG, self).__init__()
        self.input_size = 32
        self.features = self._make_layers(vgg_dict[vgg_name])
        self.n_maps = vgg_dict[vgg_name][-2]
        self.fc = self._make_fc_layers()
        self.classifier = nn.Linear(self.n_maps, num_classes)
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x, return_feat=False):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        if self.dropout is not None:
            out = self.dropout(out)
        features = self.fc(out)
        out = self.classifier(features)
        if return_feat:
            return out, features.squeeze()
        return out

    def _make_fc_layers(self):
        layers = []
        layers += [
            nn.Linear(self.n_maps * self.input_size * self.input_size, self.n_maps),
            nn.BatchNorm1d(self.n_maps),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def _make_layers(self, vgg_dict):
        layers = []
        in_channels = 3
        for x in vgg_dict:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
                self.input_size = self.input_size // 2
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        return nn.Sequential(*layers)

def make_vgg(model_name,classes):
    vgg_dict = { "VGG9": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M"],
             "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
             "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
            }
    return VGG(model_name,vgg_dict, classes)

class VGG9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VGG9, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 32 * 32, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    