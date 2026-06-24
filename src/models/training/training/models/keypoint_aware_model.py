#!/usr/bin/env python3
"""

"""

import torch
import torch.nn as nn
from torchvision import models


class KeypointAwareModel(nn.Module):
    """"""

    def __init__(self, base_model, num_classes, keypoint_dim=128):
        """

        Args:
            base_model: EfficientNetResNet
            num_classes: 
            keypoint_dim: 
        """
        super(KeypointAwareModel, self).__init__()

        # 
        self.base_model = base_model

        # 
        if hasattr(base_model, "classifier"):
            if isinstance(base_model.classifier, nn.Sequential):
                self.base_feature_dim = base_model.classifier[-1].in_features
            else:
                self.base_feature_dim = base_model.classifier.in_features
        elif hasattr(base_model, "fc"):
            self.base_feature_dim = base_model.fc.in_features
        else:
            raise ValueError("classifierfc")

        # 
        if hasattr(base_model, "classifier"):
            base_model.classifier = nn.Identity()
        elif hasattr(base_model, "fc"):
            base_model.fc = nn.Identity()

        # 
        self.keypoint_encoder = nn.Sequential(
            nn.Linear(17 * 2 * 3, 64),  # 17 * 2 * 3
            nn.ReLU(),
            nn.Linear(64, keypoint_dim),
            nn.ReLU(),
        )

        # 
        self.classifier = nn.Sequential(
            nn.Linear(self.base_feature_dim + keypoint_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x, keypoints=None):
        """

        Args:
            x: 
            keypoints: 

        Returns:
            
        """
        # 
        base_features = self.base_model(x)

        # 
        if keypoints is not None:
            # 
            keypoint_features = self._process_keypoints(keypoints)
            # 
            fused_features = torch.cat([base_features, keypoint_features], dim=1)
        else:
            # 
            fused_features = base_features

        # 
        output = self.classifier(fused_features)
        return output

    def _process_keypoints(self, keypoints):
        """

        Args:
            keypoints: 

        Returns:
            
        """
        # keypoints(batch_size, 102)
        # 34172
        # 42212
        # 34172

        # 
        keypoint_features = self.keypoint_encoder(keypoints)
        return keypoint_features


def get_keypoint_aware_model(model_type, num_classes, keypoint_dim=128):
    """

    Args:
        model_type: 
        num_classes: 
        keypoint_dim: 

    Returns:
        KeypointAwareModel
    """
    # 
    if model_type == "mobilenet_v2":
        base_model = models.mobilenet_v2(pretrained=True)
    elif model_type == "efficientnet_b0":
        base_model = models.efficientnet_b0(pretrained=True)
    elif model_type == "efficientnet_b3":
        base_model = models.efficientnet_b3(pretrained=True)
    elif model_type == "resnet50":
        base_model = models.resnet50(pretrained=True)
    else:
        raise ValueError(f": {model_type}")

    # 
    model = KeypointAwareModel(base_model, num_classes, keypoint_dim)
    return model
