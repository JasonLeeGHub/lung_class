import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class res50Encoder(nn.Module):

    def __init__(self, config):
        super(res50Encoder, self).__init__()

        # load backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        self.fc = nn.Linear(2048, config.class_num, bias=True)

        # cnn feature
        self.resnet_encoder = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                         resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.resnet_classifier = resnet.avgpool

    def forward(self, x):
        features = self.resnet_encoder(x)
        logits = self.resnet_classifier(features)
        logits = self.fc(logits.squeeze())
        return features, logits


class AttentionModule(nn.Module):

    def __init__(self, config):
        super(AttentionModule, self).__init__()
        # attention
        self.attention_layer = nn.Sequential(nn.Conv2d(2048, config.attention_map_num, kernel_size=1),
                                             nn.BatchNorm2d(config.attention_map_num),
                                             nn.ReLU(inplace=True))
        self.ReLU = nn.ReLU(inplace=True)
    def forward(self, x):
        attention_raw = self.attention_layer(x)
        attention_sum = torch.sum(attention_raw, dim=1)
        attention_sum = self.ReLU(attention_sum)
        attention_sum = torch.unsqueeze(attention_sum, dim=1)
        attention_sum = F.upsample_bilinear(attention_sum, size=(224,224))
        return attention_raw, attention_sum

