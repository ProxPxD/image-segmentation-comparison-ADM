import torch
from torch import nn, wait
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, mode='avg'):
        super(PyramidPoolingModule, self).__init__()

        self.mode = mode
        self.pool_sizes = pool_sizes
        self.pooling_layers = nn.ModuleList()

        for size in pool_sizes:
            self.pooling_layers.append(self._make_stage(in_channels, size))

    def get_prior(self, size):
        prior_map = {
            'avg': nn.AdaptiveAvgPool2d,
            'max': nn.AdaptiveMaxPool2d
        }
        if self.mode not in prior_map:
            raise ValueError(f'Unknown pooling mode: {self.mode}, use max/avg')
        return prior_map[self.mode](output_size=(size, size))

    def _make_stage(self, in_channels, size):
        prior = self.get_prior(size)

        conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Upsample(size=(360, 480))
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
    def __init__(self, in_channels, num_classes, cnn_depth=3, pool_sizes=(1, 2, 3, 6), cnn_type='resnet50'):
        super().__init__()

        if cnn_type == 'cnn':
            self.backbone = self.build_cnn(in_channels, cnn_depth)
        elif cnn_type == 'resnet34':
            resnet = nn.Sequential(*list(resnet34(weights=None).children())[:-2])
            last_seq_layer = list(resnet.children())[-1]
            last_basic_block = list(last_seq_layer.children())[-1]
            last_conv = list(last_basic_block.children())[-2]
            self.backbone = nn.Sequential(
                resnet,
                nn.Conv2d(last_conv.out_channels, in_channels, kernel_size=3)
            )
        elif cnn_type == 'resnet50':
            resnet = nn.Sequential(*list(resnet50(weights=None).children())[:-2])
            last_seq_layer = list(resnet.children())[-1]
            last_bottleneck = list(last_seq_layer.children())[-1]
            last_conv = list(last_bottleneck.children())[-3]
            self.backbone = nn.Sequential(
                resnet,
                nn.Conv2d(2048, 1024, kernel_size=1),  # Reduce channels
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspostwoe2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.AdaptiveMaxPool2d(output_size=(360, 480))
            )
        else:
            raise ValueError(f'CNN Type: {cnn_type} not supported')

        self.pyramid_pooling = PyramidPoolingModule(in_channels, pool_sizes)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_classes + len(pool_sizes), num_classes, kernel_size=1),
        )

    def build_cnn(self, in_channels, depth):
        layers = []
        for _ in range(depth):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        cnn_output = self.backbone(x)
        ppm_output = self.pyramid_pooling(cnn_output)
        final_output = self.upsample(ppm_output)
        return final_output

