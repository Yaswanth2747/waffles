# splitter.py

import torch
import torch.nn as nn


class SplitModel:

    def __init__(self, model, split_idx):
        self.model = model

        if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            self.arch = "vit"
            self.layers = list(model.encoder.layers)
            self.num_layers = len(self.layers)
        elif hasattr(model, "features"):
            self.arch = "cnn"
            features = list(model.features)
            self.num_layers = len(features)
            self.edge = nn.Sequential(*features[:split_idx])
            self.server = nn.Sequential(
                *features[split_idx:],
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                model.classifier,
            )
        else:
            raise AttributeError("Unsupported model architecture for splitting")

        if split_idx < 0 or split_idx > self.num_layers:
            raise ValueError(
                f"split_idx should be in [0, {self.num_layers}], got {split_idx}"
            )
        self.split_idx = split_idx

    def edge_forward(self, x):
        if self.arch == "cnn":
            return self.edge(x)

        x = self.model._process_input(x)
        n = x.shape[0]
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.model.encoder.pos_embedding
        x = self.model.encoder.dropout(x)
        for i in range(self.split_idx):
            x = self.layers[i](x)
        return x

    def server_forward(self, x):
        if self.arch == "cnn":
            return self.server(x)

        for i in range(self.split_idx, self.num_layers):
            x = self.layers[i](x)
        x = self.model.encoder.ln(x)
        x = x[:, 0]
        x = self.model.heads(x)
        return x
