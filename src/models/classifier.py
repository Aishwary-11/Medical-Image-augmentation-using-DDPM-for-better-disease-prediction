import torch.nn as nn
from torchvision import models

class PneumoniaClassifier(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=2, use_weights=True):
        super(PneumoniaClassifier, self).__init__()

        if model_name == 'resnet50':
            if use_weights:
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet50(weights=None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)

        elif model_name == 'densenet121':
            if use_weights:
                self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.densenet121(weights=None)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
