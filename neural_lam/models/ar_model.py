# Standard library
import os

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb

# First-party
from neural_lam import constants, metrics, utils, vis
from torchmetrics.classification import BinaryAccuracy


class ARModel(pl.LightningModule):
    """
    Generic auto-regressive weather model.
    Abstract class that can be extended.
    """

    # pylint: disable=arguments-differ
    # Disable to override args/kwargs from superclass

    def __init__(self, args, dataset_config, output_std=False):
        super().__init__()
        # Load static features for grid/data
        static_data_dict = utils.load_static_data(dataset_config.dataset)
        # print(static_data_dict.keys())
        for static_data_name, static_data_tensor in static_data_dict.items():
            self.register_buffer(
                static_data_name, static_data_tensor, persistent=False
            )

        # todo
        self.accuracy = BinaryAccuracy()

        self.step_diff_mean = torch.nan_to_num(self.step_diff_mean)
        self.step_diff_std = torch.nan_to_num(self.step_diff_std)

        # Double grid output dim. to also output std.-dev.
        self.output_std = output_std
        if self.output_std:
            self.grid_output_dim = (
                2 * constants.GRID_STATE_DIM
            )  # Pred. dim. in grid cell
        else:
            self.grid_output_dim = (
                constants.GRID_STATE_DIM
            )  # Pred. dim. in grid cell

            # Store constant per-variable std.-dev. weighting
            # Note that this is the inverse of the multiplicative weighting
            # in wMSE/wMAE
            self.register_buffer(
                "per_var_std",
                self.step_diff_std / torch.sqrt(self.param_weights),
                persistent=False,
            )

        # grid_dim from data + static
        (
            self.num_grid_nodes,
            grid_static_dim,
        ) = self.grid_static_features.shape  # 63784 = 268x238
        self.grid_dim = (
            2 * constants.GRID_STATE_DIM
            + grid_static_dim
            + constants.GRID_FORCING_DIM
        )

        # Instantiate loss function
        self.loss = metrics.get_metric(args.loss)

        self.boundary_forcing = self.border_mask is not None
        if self.boundary_forcing:
            # Pre-compute interior mask for use in loss function
            interior_mask_tensor = 1.0 - self.border_mask
        else:
            # Still set as constant 1 (no border), for plotting
            interior_mask_tensor = torch.ones(self.num_grid_nodes, 1)

        self.register_buffer(
            "interior_mask", interior_mask_tensor, persistent=False
        )  # (num_grid_nodes, 1), 1 for non-border

        # For validation and testing
        self.step_length = args.step_length  # Number of hours per pred. step
        self.val_metrics = {
            "mse": [],
        }
        self.test_metrics = {
            "mse": [],
            "mae": [],
        }
        if self.output_std:
            self.test_metrics["output_std"] = []  # Treat as metric

        # Do not try to log at lead times not forecasted
        self.val_log_leads = constants.VAL_STEP_LOG_ERRORS[
            constants.VAL_STEP_LOG_ERRORS <= args.eval_leads
        ]
        self.val_plot_vars = {
            var_i: ts[ts <= args.eval_leads]
            for var_i, ts in constants.VAL_PLOT_VARS.items()
        }
        self.eval_leads = args.eval_leads

        # For making restoring of optimizer state optional (slight hack)
        self.opt_state = None

        # For example plotting
        self.n_example_pred = args.n_example_pred
        self.plotted_examples = 0

        # For storing spatial loss maps during evaluation
        self.spatial_loss_maps = []
        self.save_hyperparameters()

        self.lr = args.lr

    def configure_optimizers(self):
        # adamw optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=0.1
        )
        if self.args.lr_scheduler.lower() == "y":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.args.lr_scheduler_factor,
                patience=self.args.lr_scheduler_patience,
                verbose=True,
            )
        else:
            scheduler = None
        if self.opt_state:
            opt.load_state_dict(self.opt_state)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_mean_loss",
                "frequency": 1,
            },
        }

        return opt

    @property
    def interior_mask_bool(self):
        """
        Get the interior mask as a boolean (N,) mask.
        If not using boundary forcing, returns None.
        """
        if not self.boundary_forcing:
            return None

        return self.interior_mask[:, 0].to(torch.bool)

    @staticmethod
    def expand_to_batch(x, batch_size):
        """
        Expand tensor with initial batch dimension
        """
        return x.unsqueeze(0).expand(batch_size, -1, -1)

    def predict_step(self, prev_state, prev_prev_state, forcing):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)
        """
        raise NotImplementedError("No prediction step implemented")

    def optional_boundary_forcing(self, pred_state, border_state):
        """
        Overwrite border with true state if using boundary forcing
        """
        if self.boundary_forcing:
            return (
                self.border_mask * border_state
                + self.interior_mask * pred_state
            )

        # Just return prediction
        return pred_state

    def unroll_prediction(self, init_states, forcing_features, true_states):
        """
        Roll out prediction taking multiple autoregressive steps with model
        init_states: (B, 2, num_grid_nodes, d_f)
        forcing_features: (B, pred_steps, num_grid_nodes, d_static_f)
        true_states: (B, pred_steps, num_grid_nodes, d_f)
        """
        prev_prev_state = init_states[:, 0]
        prev_state = init_states[:, 1]
        prediction_list = []
        pred_std_list = []
        pred_steps = forcing_features.shape[1]
        # breakpoint()

        for i in range(pred_steps):
            forcing = forcing_features[:, i]
            border_state = true_states[:, i]

            pred_state, pred_std = self.predict_step(
                prev_state, prev_prev_state, forcing
            )
            # state: (B, num_grid_nodes, d_f)
            # pred_std: (B, num_grid_nodes, d_f) or None

            new_state = self.optional_boundary_forcing(
                pred_state, border_state
            )

            prediction_list.append(new_state)
            if self.output_std:
                pred_std_list.append(pred_std)

            # Update conditioning states
            prev_prev_state = prev_state
            prev_state = new_state

        prediction = torch.stack(
            prediction_list, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        if self.output_std:
            pred_std = torch.stack(
                pred_std_list, dim=1
            )  # (B, pred_steps, num_grid_nodes, d_f)
        else:
            pred_std = self.per_var_std  # (d_f,)

        return prediction, pred_std

    def common_step(self, batch):
        """
        Predict on single batch
        batch consists of:
        init_states: (B, 2, num_grid_nodes, d_features)
        target_states: (B, pred_steps, num_grid_nodes, d_features)
        forcing_features: (B, pred_steps, num_grid_nodes, d_forcing),
            where index 0 corresponds to index 1 of init_states
        """
        # slice incase we are using a ds which return indexes
        (
            init_states,
            target_states,
            forcing_features,
        ) = batch[:3]
        # breakpoint()

        prediction, pred_std = self.unroll_prediction(
            init_states, forcing_features, target_states
        )  # (B, pred_steps, num_grid_nodes, d_f)
        # prediction: (B, pred_steps, num_grid_nodes, d_f)
        # pred_std: (B, pred_steps, num_grid_nodes, d_f) or (d_f,)

        return prediction, target_states, pred_std

    def training_step(self, batch):
        """
        Train on single batch
        """
        if self.classifier:
            pred = self.classifier_step(batch)
            target_tensor = batch[2].to(torch.bfloat16)
            pred_tensor = pred.to(torch.bfloat16)
            batch_loss = self.loss(pred_tensor, target_tensor)
            log_dict = {"train_loss": batch_loss}
            self.log_dict(
                log_dict,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            return batch_loss

        prediction, target, pred_std = self.common_step(batch)
        # breakpoint()
        # Compute loss
        batch_loss = torch.mean(
            self.loss(
                prediction,
                target,
                pred_std,
                mask=self.interior_mask_bool,
                grid_weights=self.grid_weights,
            )
        )  # mean over unrolled times and batch

        log_dict = {"train_loss": batch_loss}
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return batch_loss

    def all_gather_cat(self, tensor_to_gather):
        """
        Gather tensors across all ranks, and concatenate across dim. 0
        (instead of stacking in new dim. 0)

        tensor_to_gather: (d1, d2, ...), distributed over K ranks

        returns: (K*d1, d2, ...)
        """
        return self.all_gather(tensor_to_gather).flatten(0, 1)

    def validation_step_classifier(self, batch):
        """
        Validate on single batch
        """

        pred = self.classifier_step(batch)
        # print(pred)

        target_tensor = batch[2].to(torch.bfloat16)
        # print(target_tensor)
        pred_tensor = pred.to(torch.bfloat16)
        loss = self.loss(pred_tensor, target_tensor)
        target_tensor = target_tensor.to(torch.float32)
        pred_tensor = pred_tensor.to(torch.float32)

        accuracy = self.accuracy(pred_tensor, target_tensor)
        self.log(
            "val_acc", accuracy, on_step=False, on_epoch=True, sync_dist=True
        )
        self.log(
            "val_mean_loss", loss, on_step=False, on_epoch=True, sync_dist=True
        )

    def test_step_classifier(self, batch):
        """
        test on single batch
        """

        pred = self.classifier_step(batch)
        # print(pred)

        target_tensor = batch[2].to(torch.bfloat16)
        # print(target_tensor)
        pred_tensor = pred.to(torch.bfloat16)
        loss = self.loss(pred_tensor, target_tensor)
        target_tensor = target_tensor.to(torch.float32)
        pred_tensor = pred_tensor.to(torch.float32)

        accuracy = self.accuracy(pred_tensor, target_tensor)
        self.log(
            "test_acc", accuracy, on_step=False, on_epoch=True, sync_dist=True
        )
        self.log(
            "test_mean_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    # newer lightning versions requires batch_idx argument, even if unused
    # pylint: disable-next=unused-argument
    def validation_step(self, batch, batch_idx):
        """
        Run validation on single batch
        """
        if self.classifier:
            return self.validation_step_classifier(batch)
        prediction, target, pred_std = self.common_step(batch)
        time_step_loss = torch.mean(
            self.loss(
                prediction,
                target,
                pred_std,
                mask=self.interior_mask_bool,
                grid_weights=self.grid_weights,
            ),
            dim=0,
        )  # (time_steps-1)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        val_log_dict = {
            f"val_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.val_log_leads
        }
        mse = metrics.mse(
            prediction,
            target,
            mask=self.interior_mask_bool,
            pred_std=pred_std,
            average_grid=False,
            sum_vars=False,
        )
        for time_step in self.val_log_leads:
            val_log_dict[f"val_rmse_unroll{time_step}"] = torch.sqrt(
                torch.mean(mse[time_step - 1])
            )

        val_log_dict["val_mean_loss"] = mean_loss
        # Store MSEs
        entry_mses = metrics.mse(
            prediction,
            target,
            pred_std,
            grid_weights=self.grid_weights,
            mask=self.interior_mask_bool,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.val_metrics["mse"].append(entry_mses)

        rmse = torch.sqrt(entry_mses)

        self.log_dict(
            val_log_dict, on_step=False, on_epoch=True, sync_dist=True
        )

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch
        """
        # Create error maps for all test metrics
        if self.classifier:
            return
        self.aggregate_and_plot_metrics(self.val_metrics, prefix="val")

        # Clear lists with validation metrics values
        for metric, metric_list in self.val_metrics.items():

            metric_list.clear()

    # pylint: disable-next=unused-argument
    def test_step(self, batch, batch_idx):
        """
        Run test on single batch
        """
        if self.classifier:
            return self.test_step_classifier(batch)
        prediction, target, pred_std = self.common_step(batch)

        # prediction: (B, pred_steps, num_grid_nodes, d_f)
        # pred_std: (B, pred_steps, num_grid_nodes, d_f) or (d_f,)

        time_step_loss = torch.mean(
            self.loss(
                prediction,
                target,
                pred_std,
                mask=self.interior_mask_bool,
                grid_weights=self.grid_weights,
            ),
            dim=0,
        )  # (time_steps-1,)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        test_log_dict = {
            f"test_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.val_log_leads
        }
        test_log_dict["test_mean_loss"] = mean_loss

        self.log_dict(
            test_log_dict, on_step=False, on_epoch=True, sync_dist=True
        )

        # Compute all evaluation metrics for error maps
        # Note: explicitly list metrics here, as test_metrics can contain
        # additional ones, computed differently, but that should be aggregated
        # on_test_epoch_end
        for metric_name in ("mse", "mae"):
            metric_func = metrics.get_metric(metric_name)
            batch_metric_vals = metric_func(
                prediction,
                target,
                pred_std,
                grid_weights=self.grid_weights,
                mask=self.interior_mask_bool,
                sum_vars=False,
            )  # (B, pred_steps, d_f)
            self.test_metrics[metric_name].append(batch_metric_vals)

        if self.output_std:
            # Store output std. per variable, spatially averaged
            mean_pred_std = torch.mean(
                pred_std[..., self.interior_mask_bool, :], dim=-2
            )  # (B, pred_steps, d_f)
            self.test_metrics["output_std"].append(mean_pred_std)

        # Save per-sample spatial loss for specific times
        # Note: Do not include grid weighting for this plot
        spatial_loss = self.loss(
            prediction, target, pred_std, average_grid=False
        )  # (B, pred_steps, num_grid_nodes)
        log_spatial_losses = spatial_loss[:, self.val_log_leads - 1]
        self.spatial_loss_maps.append(log_spatial_losses)
        # (B, N_log, num_grid_nodes)

        # Plot example predictions (on rank 0 only)
        if (
            self.trainer.is_global_zero
            and self.plotted_examples < self.n_example_pred
        ):
            # Need to plot more example predictions
            n_additional_examples = min(
                prediction.shape[0],
                self.n_example_pred - self.plotted_examples,
            )

            self.plot_examples(
                batch, n_additional_examples, prediction=prediction
            )

    def plot_examples(self, batch, n_examples, prediction=None):
        """
        Plot the first n_examples forecasts from batch

        batch: batch with data to plot corresponding forecasts for
        n_examples: number of forecasts to plot
        prediction: (B, pred_steps, num_grid_nodes, d_f), existing prediction.
            Generate if None.
        """
        if prediction is None:
            prediction, target = self.common_step(batch)

        target = batch[1]

        # Rescale to original data scale
        prediction_rescaled = prediction * self.data_std + self.data_mean
        target_rescaled = target * self.data_std + self.data_mean

        # Iterate over the examples
        for pred_slice, target_slice in zip(
            prediction_rescaled[:n_examples], target_rescaled[:n_examples]
        ):
            # Each slice is (pred_steps, num_grid_nodes, d_f)
            self.plotted_examples += 1  # Increment already here

            var_vmin = (
                torch.minimum(
                    pred_slice.flatten(0, 1).min(dim=0)[0],
                    target_slice.flatten(0, 1).min(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vmax = (
                torch.maximum(
                    pred_slice.flatten(0, 1).max(dim=0)[0],
                    target_slice.flatten(0, 1).max(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vranges = list(zip(var_vmin, var_vmax))

            # Iterate over prediction horizon time steps
            for t_i, (pred_t, target_t) in enumerate(
                zip(pred_slice, target_slice), start=1
            ):
                # Create one figure per variable at this time step
                var_names = [
                    constants.PARAM_NAMES_SHORT[var_i]
                    for var_i in constants.EVAL_PLOT_VARS
                ]
                var_figs = [
                    vis.plot_prediction(
                        pred_t[:, var_i],
                        target_t[:, var_i],
                        self.interior_mask[:, 0],
                        title=(
                            f"{var_name} "
                            f"({constants.PARAM_UNITS[var_i]}), "
                            f"t={t_i} ({self.step_length*t_i} h)"
                        ),
                        vrange=var_vranges[var_i],
                    )
                    for var_i, var_name in zip(
                        constants.EVAL_PLOT_VARS, var_names
                    )
                ]

                example_i = self.plotted_examples
                wandb.log(
                    {
                        f"{var_name}_example_{example_i}": wandb.Image(fig)
                        for var_name, fig in zip(var_names, var_figs)
                    }
                )
                plt.close(
                    "all"
                )  # Close all figs for this time step, saves memory

            # Save pred and target as .pt files
            torch.save(
                pred_slice.cpu(),
                os.path.join(
                    wandb.run.dir, f"example_pred_{self.plotted_examples}.pt"
                ),
            )
            torch.save(
                target_slice.cpu(),
                os.path.join(
                    wandb.run.dir, f"example_target_{self.plotted_examples}.pt"
                ),
            )

    def create_metric_log_dict(self, metric_tensor, prefix, metric_name):
        """
        Put together a dict with everything to log for one metric.
        Also saves plots as pdf and csv if using test prefix.

        metric_tensor: (pred_steps, d_f), metric values per time and variable
        prefix: string, prefix to use for logging
        metric_name: string, name of the metric

        Return:
        log_dict: dict with everything to log for given metric
        """
        log_dict = {}
        # metric_fig = vis.plot_error_map(metric_tensor)
        full_log_name = f"{prefix}_{metric_name}"
        # log_dict[full_log_name] = wandb.Image(metric_fig)

        # if prefix == "test":
        #     # Save pdf
        #     metric_fig.savefig(
        #         os.path.join(wandb.run.dir, f"{full_log_name}.pdf")
        #     )
        #     # Save errors also as csv
        #     np.savetxt(
        #         os.path.join(wandb.run.dir, f"{full_log_name}.csv"),
        #         metric_tensor.cpu().numpy(),
        #         delimiter=",",
        #     )

        # Check if metrics are watched, log exact values for specific vars
        if full_log_name in constants.METRICS_WATCH:
            for var_i, timesteps in constants.VAR_LEADS_METRICS_WATCH.items():
                var = constants.PARAM_NAMES[var_i]
                log_dict.update(
                    {
                        f"{full_log_name}_{var}_step_{step}": metric_tensor[
                            step - 1, var_i
                        ]  # 1-indexed in constants
                        for step in timesteps
                        if step <= self.eval_leads
                    }
                )

        return log_dict

    def aggregate_and_plot_metrics(self, metrics_dict, prefix):
        """
        Aggregate and create error map plots for all metrics in metrics_dict

        metrics_dict: dictionary with metric_names and list of tensors
            with step-evals.
        prefix: string, prefix to use for logging
        """
        log_dict = {}
        for metric_name, metric_val_list in metrics_dict.items():
            metric_tensor = self.all_gather_cat(
                torch.cat(metric_val_list, dim=0)
            )  # (N_eval, pred_steps, d_f)

            if self.trainer.is_global_zero:
                metric_tensor_averaged = torch.mean(metric_tensor, dim=0)
                # (pred_steps, d_f)

                # Take square root after averaging to change squared metrics
                if "mse" in metric_name:
                    metric_tensor_averaged = torch.sqrt(metric_tensor_averaged)
                    metric_name = metric_name.replace("mse", "rmse")
                elif metric_name.endswith("_squared"):
                    metric_tensor_averaged = torch.sqrt(metric_tensor_averaged)
                    metric_name = metric_name[: -len("_squared")]

                # Note: we here assume rescaling for all metrics is linear
                metric_rescaled = metric_tensor_averaged * self.data_std
                # (pred_steps, d_f)
                log_dict.update(
                    self.create_metric_log_dict(
                        metric_rescaled, prefix, metric_name
                    )
                )

        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            wandb.log(log_dict)  # Log all
            plt.close("all")  # Close all figs

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch.
        Will gather stored tensors and perform plotting and logging on rank 0.
        """
        # Create error maps for all test metrics
        if self.classifier:
            return

        # Create error maps for all test metrics
        self.aggregate_and_plot_metrics(self.test_metrics, prefix="test")

        # Plot spatial loss maps
        spatial_loss_tensor = self.all_gather_cat(
            torch.cat(self.spatial_loss_maps, dim=0)
        )  # (N_test, N_log, num_grid_nodes)
        if self.trainer.is_global_zero:
            mean_spatial_loss = torch.mean(
                spatial_loss_tensor, dim=0
            )  # (N_log, num_grid_nodes)

            loss_map_figs = [
                vis.plot_spatial_error(
                    loss_map,
                    self.interior_mask[:, 0],
                    title=f"Test loss, t={t_i} ({self.step_length*t_i} h)",
                )
                for t_i, loss_map in zip(self.val_log_leads, mean_spatial_loss)
            ]

            # log all to same wandb key, sequentially
            for fig in loss_map_figs:
                wandb.log({"test_loss": wandb.Image(fig)})

            # also make without title and save as pdf
            pdf_loss_map_figs = [
                vis.plot_spatial_error(loss_map, self.interior_mask[:, 0])
                for loss_map in mean_spatial_loss
            ]
            pdf_loss_maps_dir = os.path.join(
                wandb.run.dir, "spatial_loss_maps"
            )
            os.makedirs(pdf_loss_maps_dir, exist_ok=True)
            for t_i, fig in zip(self.val_log_leads, pdf_loss_map_figs):
                fig.savefig(
                    os.path.join(pdf_loss_maps_dir, f"loss_t{t_i}.pdf")
                )
            # save mean spatial loss as .pt file also
            torch.save(
                mean_spatial_loss.cpu(),
                os.path.join(wandb.run.dir, "mean_spatial_loss.pt"),
            )

        self.spatial_loss_maps.clear()

    def on_load_checkpoint(self, checkpoint):
        """
        Perform any changes to state dict before loading checkpoint
        """
        loaded_state_dict = checkpoint["state_dict"]

        # Fix for loading older models after IneractionNet refactoring, where
        # the grid MLP was moved outside the encoder InteractionNet class
        if "g2m_gnn.grid_mlp.0.weight" in loaded_state_dict:
            replace_keys = list(
                filter(
                    lambda key: key.startswith("g2m_gnn.grid_mlp"),
                    loaded_state_dict.keys(),
                )
            )
            for old_key in replace_keys:
                new_key = old_key.replace(
                    "g2m_gnn.grid_mlp", "encoding_grid_mlp"
                )
                loaded_state_dict[new_key] = loaded_state_dict[old_key]
                del loaded_state_dict[old_key]
