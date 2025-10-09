from torch import nn
import torch
from torchvision.models import efficientnet_b0
import torch.nn.functional as F

from utils import GradReverse


class Feature_extractor(nn.Module):
    def __init__(self):
        super(Feature_extractor, self).__init__()
        model = efficientnet_b0(weights=None, num_classes=2)
        self.encoder = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d(output_size=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = F.relu(x)
        return x


class Label_classifier(nn.Module):
    def __init__(self):
        super(Label_classifier, self).__init__()

        self.label_classifier = nn.Sequential(
            nn.Linear(1280, 1280),
            nn.ReLU(),
            nn.Linear(1280, 2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.label_classifier(x)
        return x


class Domain_Classifier(nn.Module):
    def __init__(self):
        super(Domain_Classifier, self).__init__()

        self.domain_classifier = nn.Sequential(
            nn.Linear(1280, 1280),
            nn.LeakyReLU(0.2),
            nn.Linear(1280, 500),
            nn.LeakyReLU(0.2),
            nn.Linear(500, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x, alpha=-1):
        x = GradReverse.apply(x, alpha)
        x = self.domain_classifier(x)
        x = torch.sigmoid(x)
        return x



if __name__ == '__main__':
    rand_input = torch.ones(size=(6, 3, 224, 224))
    model = Feature_extractor()
    label_cls = Label_classifier()
    domain_cls = Domain_Classifier()

    features = model(rand_input)
    label_out = label_cls(features)
    domain_out = domain_cls(features, 0.05)

    print('label_out', label_out.shape)
    print('domain_out', domain_out.shape)










































