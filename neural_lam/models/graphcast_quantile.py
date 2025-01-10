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
import pytorch_lightning as pl


class GraphCastQuant(pl.LightningModule):
    """
    Modified GraphCast model for binary classification tasks,
    with ability to load pre-trained GraphCast weights.mv
    """

    def __init__(self, args, global_mesh_config, pretrained_path=None):
        # super().__init__(args, global_mesh_config, 0)
        super().__init__()

        # Load the pre-trained GraphCast model
        self.graphcast = GraphCast(args, global_mesh_config, 0)
        if pretrained_path:
            self.load_pretrained(pretrained_path)

        self.loss_module = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        # Freeze GraphCast parameters (optional, remove if you want to fine-tune everything)
        for param in self.graphcast.parameters():
            param.requires_grad = False

        num_timesteps = 1
        num_quantiles = 3
        output_shape = (num_timesteps, num_quantiles)

        # quant head
        self.quantifier = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(args.hidden_dim // 2, output_shape[0] * output_shape[1]),
            nn.Sigmoid(),
        )

    def load_pretrained(self, path):
        """Load pre-trained GraphCast weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.graphcast.load_state_dict(checkpoint["state_dict"], strict=False)
        print(f"Loaded pre-trained GraphCast from {path}")

    def forward(self, atmos_state, forcing):
        """
        Forward pass for classification
        """
        # Use GraphCast's encoding and processing
        self.grid_static_features = torch.nan_to_num(self.grid_static_features)
        # Create full grid node features of shape (B, num_grid_nodes, grid_dim)
        grid_features = torch.cat(
            (
                atmos_state
                forcing,
                self.expand_to_batch((self.grid_static_features), batch_size),
            ),
            dim=-1,
        )


        grid_emb = self.graphcast.grid_embedder(
            grid_features
        )  # (B, num_grid_nodes, d_h)
        g2m_emb = self.graphcast.g2m_embedder(self.graphcast.g2m_features)  # (M_g2m, d_h)
        m2g_emb = self.graphcast.m2g_embedder(self.graphcast.m2g_features)  # (M_m2g, d_h)
        mesh_emb = self.graphcast.embedd_mesh_nodes()

        # Map from grid to mesh
        mesh_emb_expanded = self.expand_to_batch(
            mesh_emb, batch_size
        )  # (B, num_mesh_nodes, d_h)
        g2m_emb_expanded = self.expand_to_batch(g2m_emb, batch_size)

        # This also splits representation into grid and mesh
        mesh_rep = self.graphcast.g2m_gnn(
            grid_emb, mesh_emb_expanded, g2m_emb_expanded
        )  # (B, num_mesh_nodes, d_h)
        # Also MLP with residual for grid representation
        grid_rep = grid_emb + self.graphcast.encoding_grid_mlp(
            grid_emb
        )  # (B, num_grid_nodes, d_h)

        # Run processor step
        mesh_rep = self.graphcast.process_step(mesh_rep)
        breakpoint()

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
