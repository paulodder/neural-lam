import os
from typing import List, Tuple

import numpy as np
import torch
import xarray as xa
from torch.utils.data import Dataset

from neural_lam import constants, utils
from bwdl.constants import DATA_DIR
from bwdl.named_configs import DatasetConfig


class ERA5Dataset(Dataset):
    """Dataset loading ERA5 from Zarr."""

    def __init__(
        self,
        dataset_config,
        pred_length: int = 40,
        split: str = "train",
        standardize: bool = True,
        expanded_test: bool = False,
    ):
        super().__init__()
        self._validate_split(split)
        variables = constants.PARAM_NAMES
        self.step_size = self._get_step_size(dataset_config.rolling_mean)
        self.pred_length = pred_length
        self.standardize = standardize
        self.dataset_config = dataset_config
        dataset_name = dataset_config.name

        fields_xds, forcing_xda = self._load_data(dataset_name)
        self.atm_vars, self.surface_vars = self._filter_variables(variables)
        # breakpoint()
        split_slice = self._get_split_slice(dataset_name, split, expanded_test)
        fields_ds_split = fields_xds.sel(time=split_slice)
        forcing_ds_split = forcing_xda.sel(time=split_slice)

        self._setup_dataset_length(split, len(fields_ds_split.coords["time"]))
        self._setup_standardization(dataset_name)
        self._setup_data_arrays(fields_ds_split, forcing_ds_split)

        self.variables = variables

    def _validate_split(self, split: str):
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown dataset split")

    def _load_data(self, dataset_name: str) -> Tuple[xa.Dataset, xa.DataArray]:
        fields_path = DATA_DIR / "datasets" / dataset_name / "fields.zarr"
        forcing_path = DATA_DIR / "datasets" / dataset_name / "forcing.zarr"
        return xa.open_zarr(fields_path), xa.open_dataarray(
            forcing_path, engine="zarr"
        )

    def _filter_variables(
        self, variables: List[str]
    ) -> Tuple[List[str], List[str]]:
        return [var for var in variables if "-" in var], [
            var for var in variables if "-" not in var
        ]

    def _get_split_slice(
        self, dataset_name: str, split: str, expanded_test: bool
    ) -> slice:
        if "example" in dataset_name:
            return self._get_example_split_slice(split)
        return self._get_actual_split_slice(split, expanded_test)

    def _get_example_split_slice(self, split: str) -> slice:
        split_slices = {
            "train": slice("1959-01-01T12", "1959-01-03T12"),
            "val": slice("1959-01-03T18", "1959-01-04T18"),
            "test": slice("1959-01-03T18", "1959-01-04T18"),
        }
        return split_slices[split]

    def _get_actual_split_slice(
        self, split: str, expanded_test: bool
    ) -> slice:
        split_slices = {
            "train": slice("1959-01-01T12", "2017-12-31T12"),
            "val": slice("2017-12-31T18", "2019-12-31T12"),
            "test": (
                slice("2019-12-31T18", "2023-12-31T18")
                if expanded_test
                else slice("2019-12-31T18", "2021-01-10T18")
            ),
        }
        return split_slices[split]

    def _setup_dataset_length(self, split: str, timesteps_in_split: int):
        ds_timesteps = (
            timesteps_in_split - 1 - self.pred_length - (2 * self.step_size)
        )
        if ds_timesteps <= 0:
            raise ValueError("Dataset too small for given pred_length")

        if split == "train":
            self.ds_len = ds_timesteps
            self.init_all = True
        else:  # val, test
            self.ds_len = int(np.ceil(ds_timesteps / 2))
            self.init_all = False

    def _setup_standardization(self, dataset_name: str):
        if self.standardize:
            ds_stats = utils.load_dataset_stats(dataset_name, "cpu")
            self.data_mean = ds_stats["data_mean"]
            self.data_std = ds_stats["data_std"]
            # selected_vars = self.atm_vars + self.surface_vars
            self.data_mean = self.data_mean
            self.data_std = self.data_std

    def _setup_data_arrays(
        self, fields_ds_split: xa.Dataset, forcing_ds_split: xa.DataArray
    ):
        self.atm_xda, self.atm_total_dim = self._setup_atm_data(
            fields_ds_split
        )
        self.surface_xda, self.surface_total_dim = self._setup_surface_data(
            fields_ds_split
        )
        self.forcing_xda = forcing_ds_split

        if self.atm_total_dim == 0 and self.surface_total_dim == 0:
            raise ValueError("No variables selected")

    def _setup_atm_data(
        self, fields_ds_split: xa.Dataset
    ) -> Tuple[xa.DataArray, int]:
        if not self.atm_vars:
            return None, 0

        atm_data_list = []
        for var in self.atm_vars:
            var_name, level = var.split("-")
            var_data = fields_ds_split[var_name].sel(level=int(level))
            atm_data_list.append(var_data)

        atm_xda = xa.concat(atm_data_list, dim="state_var").transpose(
            "time", "longitude", "latitude", "state_var"
        )
        atm_total_dim = len(atm_xda.coords["state_var"])
        return atm_xda, atm_total_dim

    def _setup_surface_data(
        self, fields_ds_split: xa.Dataset
    ) -> Tuple[xa.DataArray, int]:
        if not self.surface_vars:
            return None, 0

        surface_xda = (
            fields_ds_split[self.surface_vars]
            .to_dataarray("state_var")
            .transpose("time", "longitude", "latitude", "state_var")
        )
        surface_total_dim = len(surface_xda.coords["state_var"])
        return surface_xda, surface_total_dim

    def __len__(self) -> int:
        return self.ds_len

    def _get_step_size(self, rolling_mean: str) -> int:
        """Determine the step size based on the rolling mean."""
        if rolling_mean == "1d":
            return 1
        elif rolling_mean == "3d":
            return 3
        elif rolling_mean == "1w":
            return 7
        else:
            raise ValueError(f"Unsupported rolling mean: {rolling_mean}")

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.init_all:
            init_i = idx + self.step_size  # s = idx+1
        else:
            # Only initialize at 00/12 UTC timesteps
            init_i = self.step_size + idx * 2  # s = 1 + 2idx
        sample_slice = slice(
            init_i - self.step_size,
            init_i + ((self.pred_length + 1) * self.step_size),
            self.step_size,
        )
        full_series_len = self.pred_length + 2

        # if self.init_all:
        #     init_i = self.step_size * (idx + 1)
        # else:
        #     init_i = 1 + idx * 2 * self.step_size

        # sample_slice = slice(
        #     init_i - self.step_size,
        #     init_i + (self.pred_length + 1) * self.step_size,
        #     self.step_size,
        # )
        # full_series_len = self.pred_length + 2

        full_state_torch = self._get_full_state(sample_slice, full_series_len)
        init_states, target_states = self._split_states(full_state_torch)
        forcing_torch = self._get_forcing(sample_slice, full_series_len)

        return init_states, target_states, forcing_torch

    # def __getitem__(
    #     self, idx: int
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     init_i = idx + 1 if self.init_all else 1 + idx * 2
    #     sample_slice = slice(init_i - 1, init_i + self.pred_length + 1)
    #     full_series_len = self.pred_length + 2

    #     full_state_torch = self._get_full_state(sample_slice, full_series_len)
    #     init_states, target_states = self._split_states(full_state_torch)
    #     forcing_torch = self._get_forcing(sample_slice, full_series_len)

    #     return init_states, target_states, forcing_torch

    # def __getitem__(
    #     self, idx: int
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     init_i = idx + 1 if self.init_all else 1 + idx * 2
    #     sample_slice = slice(init_i - 1, init_i + self.pred_length + 1)
    #     full_series_len = self.pred_length + 2

    #     full_state_torch = self._get_full_state(sample_slice, full_series_len)
    #     init_states, target_states = self._split_states(full_state_torch)
    #     forcing_torch = self._get_forcing(sample_slice, full_series_len)

    #     return init_states, target_states, forcing_torch

    def _get_full_state(
        self, sample_slice: slice, full_series_len: int
    ) -> torch.Tensor:
        atm_sample_np = (
            self.atm_xda[sample_slice].to_numpy()
            if self.atm_xda is not None
            else None
        )
        surface_sample_np = (
            self.surface_xda[sample_slice].to_numpy()
            if self.surface_xda is not None
            else None
        )

        full_state_np = self._concatenate_state_data(
            atm_sample_np, surface_sample_np, full_series_len
        )
        full_state_torch = torch.tensor(full_state_np, dtype=torch.float32)

        if self.standardize:
            full_state_torch = (
                full_state_torch - self.data_mean
            ) / self.data_std

        return full_state_torch

    def _concatenate_state_data(
        self,
        atm_sample_np: np.ndarray,
        surface_sample_np: np.ndarray,
        full_series_len: int,
    ) -> np.ndarray:
        state_components = []

        if atm_sample_np is not None:
            atm_reshaped = atm_sample_np.reshape(
                (full_series_len, -1, self.atm_total_dim)
            )
            state_components.append(atm_reshaped)

        if surface_sample_np is not None:
            surface_reshaped = surface_sample_np.reshape(
                (full_series_len, -1, self.surface_total_dim)
            )
            state_components.append(surface_reshaped)

        return np.concatenate(state_components, axis=-1)

    def _split_states(
        self, full_state_torch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return full_state_torch[:2], full_state_torch[2:]

    def _get_forcing(
        self, sample_slice: slice, full_series_len: int
    ) -> torch.Tensor:
        forcing_np = self.forcing_xda[sample_slice].to_numpy()
        forcing_flat_np = forcing_np.reshape(
            full_series_len, -1, forcing_np.shape[-1]
        )
        forcing_windowed = np.concatenate(
            (forcing_flat_np[:-2], forcing_flat_np[1:-1], forcing_flat_np[2:]),
            axis=2,
        )
        return torch.tensor(forcing_windowed, dtype=torch.float32)
