import torch
import torch.nn as nn
import torch_geometric as pyg
from neural_lam import utils
from neural_lam.interaction_net import InteractionNet
from neural_lam.models.base_graph_model import BaseGraphModel
from neural_lam.models.graphcast import (
    GraphCast,
)  # Import the original GraphCast
from pytorch_forecasting.metrics.quantile import QuantileLoss


class GraphCastQuant:
    """
    Modified GraphCast model for binary classification tasks,
    with ability to load pre-trained GraphCast weights.mv
    """

    def __init__(self, args, global_mesh_config, pretrained_path=None):
        # super().__init__(args, global_mesh_config, 0)

        # Load the pre-trained GraphCast model
        self.graphcast = GraphCast(args, global_mesh_config, 0)
        if pretrained_path:
            self.load_pretrained(pretrained_path)

        self.loss_module = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        # Freeze GraphCast parameters (optional, remove if you want to fine-tune everything)
        for param in self.graphcast.parameters():
            param.requires_grad = False

        # Classification head
        self.quantifier = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(args.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def load_pretrained(self, path):
        """Load pre-trained GraphCast weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.graphcast.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded pre-trained GraphCast from {path}")

    def forward(self, grid_state, grid_forcing):
        """
        Forward pass for classification
        """
        # Use GraphCast's encoding and processing
        mesh_rep = self.graphcast.grid_to_mesh(grid_state, grid_forcing)
        mesh_rep = self.graphcast.process_step(mesh_rep)

        # Global average pooling
        mesh_rep = torch.mean(mesh_rep, dim=1)  # (B, d_h)

        # Classification
        output = self.quantifier(mesh_rep)

        return output.squeeze()

    def training_step(self, batch, batch_idx):
        """
        Training step for binary classification
        """
        grid_state, grid_forcing, target = batch
        pred = self(grid_state, grid_forcing)
        loss = self.loss_module(pred, target.float())

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for binary classification
        """
        grid_state, grid_forcing, target = batch
        pred = self(grid_state, grid_forcing)
        loss = nn.BCELoss()(pred, target.float())

        self.log(
            "val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step for binary classification
        """
        grid_state, grid_forcing, target = batch
        pred = self(grid_state, grid_forcing)
        loss = nn.BCELoss()(pred, target.float())

        self.log(
            "test_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True
        )
        return loss

    # We don't need to implement get_num_mesh, embedd_mesh_nodes, and process_step
    # as we're using these methods from the original GraphCast model
