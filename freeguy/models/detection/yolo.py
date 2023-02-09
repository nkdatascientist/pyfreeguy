

"""
    [x, y, w, h]
    
    _____________________
    |                   |
    |                   |
    |                   |
    |___________________|

# Important Points
    1. To improving the accuracy for small object's they interduce 3 types of scale's
    2. Interduce Anchor boxes 
    3. yolov3 is tech report mentioned in the paper
    4. yolov3 use same padding for all the convolution layers

# Architecture
    Image -> 
        Resnet -> 
            1. 13x13
            2. 26x26
            3. 52x52

"""

import torch 

# tuple: (out_channel, kernel_size, stride)
# list: ["block", repetation]
# str: "S" => Scale prediction 
# str: "U" => Upsampling 
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class ConvolutionBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.leaky = torch.nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [
                torch.nn.Sequential(
                    ConvolutionBlock(channels, channels // 2, kernel_size=1),
                    ConvolutionBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)
        return x


class ScalePrediction(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = torch.nn.Sequential(
            ConvolutionBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            ConvolutionBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )


class YOLOv3(torch.nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, torch.nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = torch.nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    ConvolutionBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        ConvolutionBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(torch.nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3

        return layers


# if __name__ == "__main__":
#     num_classes = 80
#     IMAGE_SIZE = 416
#     model = YOLOv3(num_classes=num_classes)
#     x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
#     out = model(x)
#     print([i.shape for i in out])
#     assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
#     assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
#     assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
#     print("Dummy Inference completed...!!!")