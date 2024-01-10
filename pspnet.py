import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, mode='avg'):
        super(PyramidPoolingModule, self).__init__()

        self.mode = mode
        self.pool_sizes = pool_sizes
        self.pooling_layers = nn.ModuleList()

        for size in pool_sizes:
            self.pooling_layers.append(self._make_stage(in_channels, size))

        
    def _make_stage(self, in_channels, size):
        if self.mode == 'avg':
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif self.mode == 'max':
            prior = nn.AdaptiveMaxPool2d(output_size=(size, size))
        else:
            raise ValueError(f'Uknown pooling mode: {self.mode}, use max/avg')

        conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.PReLU()
        )

        return nn.Sequential(prior, conv)

    def forward(self, x):
        print(f'Pooling input size: {x.shape}')
        h, w = x.size(2), x.size(3)
        pool_outs = [F.interpolate(input=layer(x), size=(h, w)) for layer in self.pooling_layers]
        x = torch.cat((*pool_outs, x), dim=1)
        print(x.shape)

        return x


class PSPNet(nn.Module):
    def __init__(self, in_channels, num_classes, cnn_depth=3, pool_sizes=(1,2,3,6), cnn_type='resnet18'):
        super().__init__()

        # Define your CNN backbone (FCN in this case)
        if cnn_type == 'cnn':
            self.backbone = self.build_cnn(in_channels, cnn_depth)
        elif cnn_type == 'resnet18':
            self.backbone = resnet18(weights=None)
        else:
            raise ValueError(f'CNN Type: {cnn_type} not supported')

        # Pyramid Pooling Module
        self.pyramid_pooling = PyramidPoolingModule(in_channels, pool_sizes)

        ## Final classification layer
        self.classification = nn.Conv2d(in_channels + len(pool_sizes), num_classes, kernel_size=1)

    def build_cnn(self, in_channels, depth):
        layers = []
        for _ in range(depth):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        # CNN Backbone
        cnn_output = self.backbone(x)

        # Pyramid Pooling Module
        ppm_output = self.pyramid_pooling(cnn_output)

        # # Final classification layer
        final_output = self.classification(ppm_output)

        return final_output

