# Third-party
import torch_geometric as pyg

# First-party
from neural_lam import utils
from neural_lam.interaction_net import InteractionNet
from neural_lam.models.base_graph_model import BaseGraphModel
from torch import nn
import torch


uclass GraphCast(BaseGraphModel):
    """
    Full graph-based model that can be used with different
    (non-hierarchical) graphs. Mainly based on GraphCast, but the model from
    Keisler (2022) is almost identical.
    """

    def __init__(self, args, global_mesh_config, output_std):
        super().__init__(args, global_mesh_config, output_std)

        assert (
            not self.hierarchical
        ), "GraphCast does not use a hierarchical mesh graph"

        # grid_dim from data + static + batch_static
        mesh_dim = self.mesh_static_features.shape[1]
        m2m_edges, m2m_dim = self.m2m_features.shape
        print(
            f"Edges in subgraphs: m2m={m2m_edges}, g2m={self.g2m_edges}, "
            f"m2g={self.m2g_edges}"
        )

        # Define sub-models
        # Feature embedders for mesh
        self.mesh_embedder = utils.make_mlp(
            [mesh_dim] + self.mlp_blueprint_end
        )
        self.m2m_embedder = utils.make_mlp([m2m_dim] + self.mlp_blueprint_end)

        # GNNs
        # processor
        processor_nets = [
            InteractionNet(
                self.m2m_edge_index,
                args.hidden_dim,
                hidden_layers=args.hidden_layers,
                aggr=args.mesh_aggr,
            )
            for _ in range(args.processor_layers)
        ]
        self.processor = pyg.nn.Sequential(
            "mesh_rep, edge_rep",
            [
                (net, "mesh_rep, mesh_rep, edge_rep -> mesh_rep, edge_rep")
                for net in processor_nets
            ],
        )
        self.classifier = args.classifier
        if self.classifier:
            self.pooled_mesh_size = self.num_mesh_nodes // 4
            if self.num_mesh_nodes % 4 != 0:
                self.pooled_mesh_size += 1  # Adjust for any remainder

            # Modify classifier module
            self.classifier_module = nn.Sequential(
                nn.Linear(
                    self.pooled_mesh_size * args.hidden_dim, args.hidden_dim
                ),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, args.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(args.hidden_dim // 2, 1),
            )
            # self.classifier_module = nn.Sequential(
            #     nn.Linear(args.hidden_dim * 2, args.hidden_dim),
            #     nn.ReLU(),
            #     nn.Linear(args.hidden_dim // 4, 1),
            # )
            # self.classifier_module = nn.Sequential(
            #     nn.Linear(args.hidden_dim // 2, args.hidden_dim // 4),
            #     nn.ReLU(),
            #     nn.Linear(args.hidden_dim // 4, 1),
            #     nn.Sigmoid(),
            # )
            if args.freeze:
                self.freeze_all_except_classifier_and_pooling()
                # self.print_trainable_parameters()

    def print_trainable_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name} is trainable")
            else:
                print(f"{name} is frozen")

    def set_requires_grad(self, model, requires_grad=True):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def freeze_all_except_classifier_and_pooling(self):
        # Freeze all parameters
        self.set_requires_grad(self, False)

        # Unfreeze LearnableSpatialPooling
        if hasattr(self, "spatial_pooling"):
            self.set_requires_grad(self.spatial_pooling, True)

        # Unfreeze classifier module
        if hasattr(self, "classifier_module"):
            self.set_requires_grad(self.classifier_module, True)

    def get_num_mesh(self):
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        return self.mesh_static_features.shape[0], 0

    def embedd_mesh_nodes(self):
        """
        Embed static mesh features
        Returns tensor of shape (N_mesh, d_h)
        """
        return self.mesh_embedder(self.mesh_static_features)  # (N_mesh, d_h)

    def process_step(self, mesh_rep):
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, N_mesh, d_h)
        Returns mesh_rep: (B, N_mesh, d_h)
        """
        # Embed m2m here first
        batch_size = mesh_rep.shape[0]
        m2m_emb = self.m2m_embedder(self.m2m_features)  # (M_mesh, d_h)
        m2m_emb_expanded = self.expand_to_batch(
            m2m_emb, batch_size
        )  # (B, M_mesh, d_h)

        mesh_rep, _ = self.processor(
            mesh_rep, m2m_emb_expanded
        )  # (B, N_mesh, d_h)
        return mesh_rep
