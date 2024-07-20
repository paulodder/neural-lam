# Third-party
import matplotlib.pyplot as plt
import torch
import wandb
from torch import nn

# First-party
from neural_lam import constants, utils, vis
from neural_lam.interaction_net import InteractionNet, PropagationNet
from neural_lam.models.ar_model import ARModel

import torch.nn.functional as F


class BaseGraphModel(ARModel):
    """
    Base (abstract) class for graph-based models building on
    the encode-process-decode idea.
    """

    def __init__(self, args, global_mesh_config):
        super().__init__(args)

        assert (
            args.eval is None or args.n_example_pred <= args.batch_size
        ), "Can not plot more examples than batch size during validation"

        # Load graph with static features
        # NOTE: (IMPORTANT!) mesh nodes MUST have the first
        # num_mesh_nodes indices,
        self.hierarchical, graph_ldict = utils.load_graph(global_mesh_config)
        for name, attr_value in graph_ldict.items():
            # Make BufferLists module members and register tensors as buffers
            if isinstance(attr_value, torch.Tensor):
                self.register_buffer(name, attr_value, persistent=False)
            else:
                setattr(self, name, attr_value)

        # Specify dimensions of data
        self.num_mesh_nodes, _ = self.get_num_mesh()
        print(
            f"Loaded graph with {self.num_grid_nodes + self.num_mesh_nodes} "
            f"nodes ({self.num_grid_nodes} grid, {self.num_mesh_nodes} mesh)"
        )

        # grid_dim from data + static
        self.g2m_edges, g2m_dim = self.g2m_features.shape
        self.m2g_edges, m2g_dim = self.m2g_features.shape

        # Define sub-models
        # Feature embedders for grid
        self.mlp_blueprint_end = [args.hidden_dim] * (args.hidden_layers + 1)
        self.grid_embedder = utils.make_mlp(
            [self.grid_dim] + self.mlp_blueprint_end
        )
        # print([self.grid_dim] + self.mlp_blueprint_end)
        self.g2m_embedder = utils.make_mlp([g2m_dim] + self.mlp_blueprint_end)
        self.m2g_embedder = utils.make_mlp([m2g_dim] + self.mlp_blueprint_end)

        # GNNs
        gnn_class = (
            PropagationNet if args.vertical_propnets else InteractionNet
        )
        # encoder
        self.g2m_gnn = gnn_class(
            self.g2m_edge_index,
            args.hidden_dim,
            hidden_layers=args.hidden_layers,
            update_edges=False,
        )
        self.encoding_grid_mlp = utils.make_mlp(
            [args.hidden_dim] + self.mlp_blueprint_end
        )

        # decoder
        self.m2g_gnn = gnn_class(
            self.m2g_edge_index,
            args.hidden_dim,
            hidden_layers=args.hidden_layers,
            update_edges=False,
        )

        # Output mapping (hidden_dim -> output_dim)
        self.output_map = utils.make_mlp(
            [args.hidden_dim] * (args.hidden_layers + 1)
            + [self.grid_output_dim],
            layer_norm=False,
        )  # No layer norm on this one

    def get_num_mesh(self):
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        raise NotImplementedError("get_num_mesh not implemented")

    def embedd_mesh_nodes(self):
        """
        Embed static mesh features
        Returns tensor of shape (num_mesh_nodes, d_h)
        """
        raise NotImplementedError("embedd_mesh_nodes not implemented")

    def process_step(self, mesh_rep):
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, num_mesh_nodes, d_h)
        Returns mesh_rep: (B, num_mesh_nodes, d_h)
        """
        raise NotImplementedError("process_step not implemented")

    # def predict_step(self, prev_state, prev_prev_state, forcing):
    #     """
    #     Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
    #     prev_state: (B, num_grid_nodes, feature_dim), X_t
    #     prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
    #     forcing: (B, num_grid_nodes, forcing_dim)
    #     """
    #     batch_size = prev_state.shape[0]
    #     print("Input NaN check:")
    #     print("prev_state NaNs:", torch.isnan(prev_state).any())
    #     print("prev_prev_state NaNs:", torch.isnan(prev_prev_state).any())
    #     print("forcing NaNs:", torch.isnan(forcing).any())
    #     self.grid_static_features = torch.nan_to_num(self.grid_static_features)
    #     print(
    #         "self.grid_static_features NaNs:",
    #         torch.isnan(self.grid_static_features).any(),
    #     )
    #     # Create full grid node features of shape (B, num_grid_nodes, grid_dim)
    #     grid_features = torch.cat(
    #         (
    #             prev_state,
    #             prev_prev_state,
    #             forcing,
    #             self.expand_to_batch(self.grid_static_features, batch_size),
    #         ),
    #         dim=-1,
    #     )

    #     print("grid_features NaNs:", torch.isnan(grid_features).any())

    #     # # Embed all features
    #     # grid_emb = self.grid_embedder(
    #     #     grid_features
    #     # )  # (B, num_grid_nodes, d_h)
    #     print("grid_features stats:")
    #     print(f"  mean: {grid_features.mean():.4f}")
    #     print(f"  std: {grid_features.std():.4f}")
    #     print(f"  min: {grid_features.min():.4f}")
    #     print(f"  max: {grid_features.max():.4f}")

    #     # Embed all features
    #     grid_emb = self.grid_embedder(grid_features)

    #     print("grid_emb stats:")
    #     print(f"  mean: {grid_emb.mean():.4f}")
    #     print(f"  std: {grid_emb.std():.4f}")
    #     print(f"  min: {grid_emb.min():.4f}")
    #     print(f"  max: {grid_emb.max():.4f}")
    #     print("grid_emb NaNs:", torch.isnan(grid_emb).any())

    #     g2m_emb = self.g2m_embedder(self.g2m_features)  # (M_g2m, d_h)
    #     print("g2m_emb NaNs:", torch.isnan(g2m_emb).any())

    #     m2g_emb = self.m2g_embedder(self.m2g_features)  # (M_m2g, d_h)
    #     print("m2g_emb NaNs:", torch.isnan(m2g_emb).any())

    #     mesh_emb = self.embedd_mesh_nodes()
    #     print("mesh_emb NaNs:", torch.isnan(mesh_emb).any())

    #     # Map from grid to mesh
    #     mesh_emb_expanded = self.expand_to_batch(
    #         mesh_emb, batch_size
    #     )  # (B, num_mesh_nodes, d_h)
    #     print("mesh_emb_expanded NaNs:", torch.isnan(mesh_emb_expanded).any())

    #     g2m_emb_expanded = self.expand_to_batch(g2m_emb, batch_size)
    #     print("g2m_emb_expanded NaNs:", torch.isnan(g2m_emb_expanded).any())

    #     # This also splits representation into grid and mesh
    #     mesh_rep = self.g2m_gnn(
    #         grid_emb, mesh_emb_expanded, g2m_emb_expanded
    #     )  # (B, num_mesh_nodes, d_h)
    #     print("mesh_rep NaNs:", torch.isnan(mesh_rep).any())

    #     # Also MLP with residual for grid representation
    #     grid_rep = grid_emb + self.encoding_grid_mlp(
    #         grid_emb
    #     )  # (B, num_grid_nodes, d_h)
    #     print("grid_rep NaNs:", torch.isnan(grid_rep).any())

    #     # Run processor step
    #     mesh_rep = self.process_step(mesh_rep)
    #     print("processed mesh_rep NaNs:", torch.isnan(mesh_rep).any())

    #     # Map back from mesh to grid
    #     m2g_emb_expanded = self.expand_to_batch(m2g_emb, batch_size)
    #     print("m2g_emb_expanded NaNs:", torch.isnan(m2g_emb_expanded).any())

    #     grid_rep = self.m2g_gnn(
    #         mesh_rep, grid_rep, m2g_emb_expanded
    #     )  # (B, num_grid_nodes, d_h)
    #     print("updated grid_rep NaNs:", torch.isnan(grid_rep).any())

    #     # Map to output dimension, only for grid
    #     net_output = self.output_map(
    #         grid_rep
    #     )  # (B, num_grid_nodes, d_grid_out)
    #     print("net_output NaNs:", torch.isnan(net_output).any())

    #     if self.output_std:
    #         pred_delta_mean, pred_std_raw = net_output.chunk(
    #             2, dim=-1
    #         )  # both (B, num_grid_nodes, d_f)
    #         print("pred_delta_mean NaNs:", torch.isnan(pred_delta_mean).any())
    #         print("pred_std_raw NaNs:", torch.isnan(pred_std_raw).any())

    #         # Note: The predicted std. is not scaled in any way here
    #         # linter for some reason does not think softplus is callable
    #         # pylint: disable-next=not-callable
    #         pred_std = torch.nn.functional.softplus(pred_std_raw)
    #         print("pred_std NaNs:", torch.isnan(pred_std).any())
    #     else:
    #         pred_delta_mean = net_output
    #         pred_std = None
    #         print("pred_delta_mean NaNs:", torch.isnan(pred_delta_mean).any())

    #     # Rescale with one-step difference statistics
    #     rescaled_delta_mean = (
    #         pred_delta_mean * self.step_diff_std + self.step_diff_mean
    #     )
    #     print(
    #         "rescaled_delta_mean NaNs:", torch.isnan(rescaled_delta_mean).any()
    #     )

    #     # Residual connection for full state
    #     final_output = prev_state + rescaled_delta_mean
    #     print("final_output NaNs:", torch.isnan(final_output).any())

    #     return final_output, pred_std

    def predict_step(self, prev_state, prev_prev_state, forcing):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)
        """
        batch_size = prev_state.shape[0]
        # breakpoint()
        self.grid_static_features = torch.nan_to_num(self.grid_static_features)
        # Create full grid node features of shape (B, num_grid_nodes, grid_dim)
        grid_features = torch.cat(
            (
                prev_state,
                prev_prev_state,
                forcing,
                self.expand_to_batch((self.grid_static_features), batch_size),
            ),
            dim=-1,
        )

        # Embed all featupres
        # breakpoint()
        grid_emb = self.grid_embedder(
            grid_features
        )  # (B, num_grid_nodes, d_h)
        g2m_emb = self.g2m_embedder(self.g2m_features)  # (M_g2m, d_h)
        m2g_emb = self.m2g_embedder(self.m2g_features)  # (M_m2g, d_h)
        mesh_emb = self.embedd_mesh_nodes()

        # Map from grid to mesh
        mesh_emb_expanded = self.expand_to_batch(
            mesh_emb, batch_size
        )  # (B, num_mesh_nodes, d_h)
        g2m_emb_expanded = self.expand_to_batch(g2m_emb, batch_size)

        # This also splits representation into grid and mesh
        mesh_rep = self.g2m_gnn(
            grid_emb, mesh_emb_expanded, g2m_emb_expanded
        )  # (B, num_mesh_nodes, d_h)
        # Also MLP with residual for grid representation
        grid_rep = grid_emb + self.encoding_grid_mlp(
            grid_emb
        )  # (B, num_grid_nodes, d_h)

        # Run processor step
        mesh_rep = self.process_step(mesh_rep)

        # Map back from mesh to grid
        m2g_emb_expanded = self.expand_to_batch(m2g_emb, batch_size)
        grid_rep = self.m2g_gnn(
            mesh_rep, grid_rep, m2g_emb_expanded
        )  # (B, num_grid_nodes, d_h)

        # Map to output dimension, only for grid
        net_output = self.output_map(
            grid_rep
        )  # (B, num_grid_nodes, d_grid_out)

        if self.output_std:
            pred_delta_mean, pred_std_raw = net_output.chunk(
                2, dim=-1
            )  # both (B, num_grid_nodes, d_f)
            # Note: The predicted std. is not scaled in any way here
            # linter for some reason does not think softplus is callable
            # pylint: disable-next=not-callable
            pred_std = torch.nn.functional.softplus(pred_std_raw)
        else:
            pred_delta_mean = net_output
            pred_std = None

        # Rescale with one-step difference statistics
        rescaled_delta_mean = (
            pred_delta_mean * self.step_diff_std + self.step_diff_mean
        )

        # Residual connection for full state
        return prev_state + rescaled_delta_mean, pred_std

    def classifier_step(self, batch):
        init_states, forcing_features, target = batch[:3]
        prev_prev_state = init_states[:, 0]
        prev_state = init_states[:, 1]
        forcing = forcing_features[:, 0]
        batch_size = prev_state.shape[0]

        self.grid_static_features = torch.nan_to_num(self.grid_static_features)
        grid_features = torch.cat(
            (
                prev_state,
                prev_prev_state,
                forcing,
                self.expand_to_batch((self.grid_static_features), batch_size),
            ),
            dim=-1,
        )

        grid_emb = self.grid_embedder(grid_features)
        g2m_emb = self.g2m_embedder(self.g2m_features)
        m2g_emb = self.m2g_embedder(self.m2g_features)
        mesh_emb = self.embedd_mesh_nodes()

        mesh_emb_expanded = self.expand_to_batch(mesh_emb, batch_size)
        g2m_emb_expanded = self.expand_to_batch(g2m_emb, batch_size)

        mesh_rep = self.g2m_gnn(grid_emb, mesh_emb_expanded, g2m_emb_expanded)
        grid_rep = grid_emb + self.encoding_grid_mlp(grid_emb)

        mesh_rep = self.process_step(mesh_rep)

        # Apply average pooling to downsample by a factor of 4
        torch.use_deterministic_algorithms(False, warn_only=True)
        pooled_mesh_rep = F.adaptive_avg_pool1d(
            mesh_rep.transpose(1, 2), self.pooled_mesh_size
        ).transpose(1, 2)
        torch.use_deterministic_algorithms(True, warn_only=True)

        # Flatten the pooled representation
        flattened_rep = pooled_mesh_rep.reshape(batch_size, -1)

        # Classification
        output = self.classifier_module(flattened_rep)

        return output[:, 0]

    def validation_step(self, batch, *args):
        """
        Run validation on single batch
        """
        super().validation_step(batch, *args)
        batch_idx = args[0]

        if self.classifier:
            return
        # Plot some example predictions
        if (
            self.trainer.is_global_zero
            and batch_idx == 0
            and self.n_example_pred > 0
        ):
            prediction, target, _ = self.common_step(batch)

            # Rescale to original data scale
            prediction_rescaled = prediction * self.data_std + self.data_mean
            target_rescaled = target * self.data_std + self.data_mean

            # Plot samples
            log_plot_dict = {}
            for example_i, (pred_traj, target_traj) in enumerate(
                zip(
                    prediction_rescaled[: self.n_example_pred],
                    target_rescaled[: self.n_example_pred],
                ),
                start=1,
            ):

                for var_i, timesteps in self.val_plot_vars.items():
                    pass
                    # var_name = constants.PARAM_NAMES[var_i]
                    # var_unit = constants.PARAM_UNITS[var_i]
                    # for step in timesteps:
                    #     pred_state = pred_traj[step - 1, :, var_i]
                    #     target_state = target_traj[step - 1, :, var_i]
                    #     # both (num_grid_nodes,)
                    #     plot_title = (
                    #         f"{var_name} ({var_unit}), t={step} "
                    #         f"({self.step_length*step} h)"
                    #     )

                    # Make plots
                    # log_plot_dict[
                    #     f"{var_name}_step_{step}_ex{example_i}"
                    # ] = vis.plot_prediction(
                    #     pred_state,
                    #     target_state,
                    #     self.interior_mask[:, 0],
                    #     title=plot_title,
                    # )

            if not self.trainer.sanity_checking:
                # Log all plots to wandb
                wandb.log(log_plot_dict)

        plt.close("all")
