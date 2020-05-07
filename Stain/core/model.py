import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

EPSILON = 1e-12
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, dim=-1)
        return feature_matrix

class res50EncoderOnly(nn.Module):
    def __init__(self, config):
        super(res50EncoderOnly, self).__init__()
        # load backbone and optimize its architecture
        resnet = torchvision.models.resnet18(pretrained=True)
        self.fc = nn.Linear(512, config.class_num, bias=False)

        # cnn feature
        self.resnet_encoder = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                         resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        # self.encoder_ = nn.Sequential()

        self.GAP = nn.AdaptiveAvgPool2d(1)
    def forward(self,x):
        x = self.resnet_encoder(x)
        x = self.GAP(x)
        x = self.fc(x.squeeze())
        return x


class res50Encoder(nn.Module):
    def __init__(self, config):
        super(res50Encoder, self).__init__()

        # load backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        self.fc = nn.Linear(2048*config.attention_map_num, config.class_num, bias=False)
        self.fc_bone = nn.Linear(2048, config.class_num, bias=True)

        # cnn feature
        self.resnet_encoder = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                         resnet.layer1, resnet.layer2, resnet.layer3)
        self.resnet_encoder4 = resnet.layer4
        self.bap = BAP(pool='GAP')
        self.attention_module = AttentionModule(config)
        # self.attention_module = SimpleAttentionModule(config)

        self.M = config.attention_map_num
        self.GAP = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x, training=True):
        batch_size = x.size(0)

        features_2 = self.resnet_encoder(x)
        features_1 = self.resnet_encoder4(features_2)
        attention_maps = self.attention_module(features_2, features_1)
        # attention_maps = self.attention_module( features_1)

        feature_matrix = self.bap(features_1, attention_maps)
        logits = self.fc(feature_matrix*100)
        # logits_bone = self.fc_bone(torch.squeeze(self.GAP(features_1)))
        # attention map 4 augment
        if training:
            # Randomly choose one of attention maps Ak
            attention_map = []
            for i in range(batch_size):
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + EPSILON)
                attention_weights = F.normalize(attention_weights, p=1, dim=0)
                k_index = np.random.choice(self.M, 3, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)  # (B, 3, H, W) -3 types of augs
        else:
            # Object Localization Am = mean(Ak)
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)

        return logits, features_1, attention_map

class SimpleAttentionModule(nn.Module):
    def __init__(self, config):
        super(SimpleAttentionModule, self).__init__()
        # attention
        self.attention_layer = nn.Sequential(nn.Conv2d(2048, config.attention_map_num, kernel_size=1),
                                             nn.BatchNorm2d(config.attention_map_num, eps=0.001),
                                             nn.ReLU(inplace=True))

    def forward(self, x):
        return self.attention_layer(x)


class AttentionModule(nn.Module):

    def __init__(self, config):
        super(AttentionModule, self).__init__()
        self.pixel_shuffel_upsample = nn.PixelShuffle(2)
        self.pixel_shuffel_upsample2 = nn.PixelShuffle(2)
        self.ReLU = nn.ReLU(inplace=True)
        self.attention_texture = nn.Sequential(nn.Conv2d(1024, 32, kernel_size=1, stride=1, bias=False),
                                             nn.BatchNorm2d(config.attention_map_num, eps=0.001),
                                             nn.ReLU(inplace=True))
        self.attention_target = nn.Sequential(nn.Conv2d(2048, 32, kernel_size=1, stride=1, bias=False),
                                             nn.BatchNorm2d(config.attention_map_num, eps=0.001),
                                             nn.ReLU(inplace=True))
        self.avgpool = nn.AvgPool2d(kernel_size=2,stride=2)
    def forward(self, x2, x1):
        # print(x.size())
        target_map = self.attention_target(x1)  # 32 channels, size
        # up2 = self.pixel_shuffel_upsample(x1) # 512 chs, size*2
        texture_map = self.attention_texture(x2)
        # attention_output = texture_map + F.interpolate(target_map, scale_factor=2, mode='bilinear')
        attention_output = target_map + self.avgpool(texture_map)
        return attention_output
