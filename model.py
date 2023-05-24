import torch
from torch import nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from sentence_transformers import SentenceTransformer


class SentenceAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        mid_dim = in_dim * 4
        self.attention = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, out_dim),
            nn.Softmax(-1)
        )

    def forward(self, image, text):
        attention = self.attention(text)
        image = image * attention.unsqueeze(-1).unsqueeze(-1)
        return image


def resize(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class MergeBlock(nn.Module):
    def __init__(self, n_channels_low, n_channels_high):
        super().__init__()
        self.low_res_conv = nn.Sequential(
            nn.Conv2d(n_channels_low, n_channels_high, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_channels_high)
        )

    def forward(self, x):
        x_low_res = x.pop()
        x_high_res = x[-1]
        x_low_res = self.low_res_conv(x_low_res)
        x_low_res = resize(x_low_res, size=x_high_res.shape[2:])
        x[-1] = torch.relu_(x_low_res + x_high_res)


class Model(nn.Module):
    def __init__(self, n_classes, dropout, backbone_name, backbone_file, encode_dim):
        super().__init__()

        self.init_norm = nn.InstanceNorm2d(3, track_running_stats=True)
        self.backbone, n_channels = self._create_backbone(backbone_name, backbone_file)

        self.sentence_attentions = nn.ModuleList([
            SentenceAttention(encode_dim, n_channels[i]) for i in range(len(n_channels))
        ])

        self.merge_blocks = nn.ModuleList([
            MergeBlock(n_channels[i], n_channels[i-1]) for i in range(1, len(n_channels))
        ])

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Conv2d(n_channels[0], n_classes, kernel_size=1)

    def _create_backbone(self, model_name, file_path):
        def extract_features(self, x):
            x = self._swish(self._bn0(self._conv_stem(x)))

            features = []
            for idx, block in enumerate(self._blocks):
                stride = block._block_args.stride
                stride = stride[0] if isinstance(stride, list) else stride
                if stride == 2:
                    features.append(x)

                drop_connect_rate = self._global_params.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / len(self._blocks)
                x = block(x, drop_connect_rate=drop_connect_rate)

            features.append(x)
            return features

        EfficientNet.extract_features = extract_features
        model = EfficientNet.from_name(model_name, image_size=None, drop_connect_rate=None)
        model_state = torch.load(file_path, map_location='cpu')
        model.load_state_dict(model_state)

        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc

        model_channels = {
            'efficientnet-b0': [24, 24, 40, 112, 320],
            'efficientnet-b1': [24, 24, 40, 112, 320],
            'efficientnet-b2': [24, 24, 48, 120, 352],
            'efficientnet-b3': [24, 32, 48, 136, 384],
            'efficientnet-b4': [24, 32, 56, 160, 448],
            'efficientnet-b5': [24, 40, 64, 176, 512],
            'efficientnet-b6': [32, 40, 72, 200, 576],
            'efficientnet-b7': [32, 48, 80, 224, 640]
        }

        return model, model_channels[model_name]

    def forward(self, x, t):
        input_size = x.shape[2:]

        x = self.init_norm(x)
        x = self.backbone.extract_features(x)

        for i in range(len(x)):
            x[i] = self.sentence_attentions[i](x[i], t)

        for i in range(len(x)-2, -1, -1):
            self.merge_blocks[i](x)
        x = x[0]

        x = self.dropout(x)
        x = self.classifier(x)
        x = resize(x, input_size)
        return x


def create_model(n_classes):
    dropout = 0.5
    backbone_name = 'efficientnet-b7'
    backbone_file = f'/path/to/pretrained/model/{backbone_name}.pth'

    sentence_encoder = 'all-MiniLM-L6-v2'
    encode_dim = 384
    model = Model(n_classes, dropout, backbone_name, backbone_file, encode_dim)

    sentence_encoder = SentenceTransformer(sentence_encoder)
    sentence_encoder.eval()
    return model, sentence_encoder
