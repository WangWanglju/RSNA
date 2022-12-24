import sys

import timm
import torch
import torch.nn as nn
from utils import load_model_weights
 
def define_model(
    cfg,
    num_classes=1,
    num_classes_aux=0,
    n_channels=1,
    pretrained_weights="",
    pretrained=True,
):
    """
    Loads a pretrained model & builds the architecture.
    Supports timm models.

    Args:
        name (str): Model name
        num_classes (int, optional): Number of classes. Defaults to 1.
        num_classes_aux (int, optional): Number of aux classes. Defaults to 0.
        n_channels (int, optional): Number of image channels. Defaults to 3.
        pretrained_weights (str, optional): Path to pretrained encoder weights. Defaults to ''.
        pretrained (bool, optional): Whether to load timm pretrained weights.

    Returns:
        torch model -- Pretrained model.
    """
    # Load pretrained model
    encoder = getattr(timm.models, cfg.model)(pretrained=pretrained)
    encoder.name = cfg.model

    # Tile Model
    model = ClsModel(
        encoder,
        num_classes=num_classes,
        num_classes_aux=num_classes_aux,
        n_channels=n_channels,
    )

    if pretrained_weights:
        model = load_model_weights(model, pretrained_weights, verbose=1, strict=False)

    return model


class ClsModel(nn.Module):
    """
    Model with an attention mechanism.
    """
    def __init__(
        self,
        encoder,
        num_classes=1,
        num_classes_aux=0,
        n_channels=3,
    ):
        """
        Constructor.

        Args:
            encoder (timm model): Encoder.
            num_classes (int, optional): Number of classes. Defaults to 1.
            num_classes_aux (int, optional): Number of aux classes. Defaults to 0.
            n_channels (int, optional): Number of image channels. Defaults to 3.
        """
        super().__init__()

        self.encoder = encoder
        self.nb_ft = encoder.num_features

        self.num_classes = num_classes
        self.num_classes_aux = num_classes_aux
        self.n_channels = n_channels

        # self.pool = nn.AdaptiveAvgPool2d(1)
        # self.features = nn.Sequential(
        #     nn.Linear(self.nb_ft, 512),
        #     nn.Dropout1d(p=0.3),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(512),
        #     )

        self.logits = nn.Linear(self.nb_ft, num_classes)
        if self.num_classes_aux:
            self.logits_aux = nn.Linear(self.nb_ft, num_classes_aux)

        self._update_num_channels()

    def _update_num_channels(self):
        if self.n_channels != 3:
            for n, m in self.encoder.named_modules():
                if n:
                    # print("Replacing", n)
                    old_conv = getattr(self.encoder, n)
                    new_conv = nn.Conv2d(
                        self.n_channels,
                        old_conv.out_channels,
                        kernel_size=old_conv.kernel_size,
                        stride=old_conv.stride,
                        padding=old_conv.padding,
                        bias=old_conv.bias is not None,
                    )
                    setattr(self.encoder, n, new_conv)
                    break

    def extract_features(self, x):
        """
        Extract features function.

        Args:
            x (torch tensor [batch_size x 3 x w x h]): Input batch.

        Returns:
            torch tensor [batch_size x num_features]: Features.
        """
        fts = self.encoder.forward_features(x)
        b, n, _, _ =fts.shape 
        # fts = self.pool(fts).reshape(b,n)
        while len(fts.size()) > 2:
            fts = fts.mean(-1)
        # fts = self.features(fts)
        return fts

    def get_logits(self, fts):
        """
        Computes logits.

        Args:
            fts (torch tensor [batch_size x num_features]): Features.

        Returns:
            torch tensor [batch_size x num_classes]: logits.
            torch tensor [batch_size x num_classes_aux]: logits aux.
        """
        
        logits = self.logits(fts)

        if self.num_classes_aux:
            logits_aux = self.logits_aux(fts)
        else:
            logits_aux = torch.zeros((fts.size(0)))

        return logits, logits_aux

    def forward(self, x, return_fts=False):
        """
        Forward function.

        Args:
            x (torch tensor [batch_size x n_frames x h x w]): Input batch.

        Returns:
            torch tensor [batch_size x num_classes]: logits.
            torch tensor [batch_size x num_classes_aux]: logits aux.
        """
        fts = self.extract_features(x)

        logits, logits_aux = self.get_logits(fts)

        return logits, logits_aux