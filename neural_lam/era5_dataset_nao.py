import numpy as np
import torch
import xarray as xa
from typing import Tuple

from neural_lam import constants
from bwdl.constants import DATA_DIR
from bwdl.named_configs import DatasetConfig
from neural_lam.era5_dataset import ERA5Dataset


class ERA5NAODataset(ERA5Dataset):
    """Dataset loading ERA5 from Zarr or NetCDF for NAO index prediction with exact date matching."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        split_config: DatasetConfig,
        lead_time: int = 1,
        split: str = "train",
        standardize: bool = True,
        expanded_test: bool = False,
        format: str = "zarr",
        return_idxs: bool = False,
    ):
        super().__init__(
            dataset_config,
            split_config,
            pred_length=1,
            split=split,
            standardize=standardize,
            expanded_test=expanded_test,
            format=format,
            return_idxs=return_idxs,
        )
        self.lead_time = lead_time
        self.nao_xda = self._load_nao_data(dataset_config.name)
        self.nao_ds_split = self.nao_xda.sel(
            time=self._get_split_slice(
                dataset_config.name, split, expanded_test
            )
        )

        # Align ERA5 and NAO index dates
        self._align_dates()

    def _load_nao_data(self, dataset_name: str) -> xa.DataArray:
        nao_path = DATA_DIR / "nao_index_all.nc"
        return xa.open_dataarray(nao_path)

    def _align_dates(self):
        # Get the common date range
        era5_dates = self.atm_xda.time.values
        nao_dates = self.nao_xda.time.values
        common_start = max(era5_dates[0], nao_dates[0])
        common_end = min(era5_dates[-1], nao_dates[-1])

        # Adjust the dataset to use only the common date range
        self.atm_xda = self.atm_xda.sel(time=slice(common_start, common_end))
        self.surface_xda = self.surface_xda.sel(
            time=slice(common_start, common_end)
        )
        self.forcing_xda = self.forcing_xda.sel(
            time=slice(common_start, common_end)
        )
        self.nao_ds_split = self.nao_ds_split.sel(
            time=slice(common_start, common_end)
        )

        # Adjust dataset length

        self.ds_len = len(self.atm_xda.time) - self.lead_time - 1
        # print("Dataset length adjusted to:", self.ds_len)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_item = super().__getitem__(idx)
        init_states = original_item[0]  # This already contains t-1 and t
        forcing = original_item[2]  # Get forcing for t-1 and t

        # Get the NAO index target
        target = self._get_nao_target(idx)

        if self.return_idxs:
            return init_states, forcing, target, idx
        return init_states, forcing, target

    def _get_nao_target(self, idx: int) -> torch.Tensor:
        sample_slice = self.get_sample_slice(idx)
        era5_date = self.atm_xda.time[sample_slice].values[2]
        target_date = era5_date + np.timedelta64(self.lead_time, "D")

        try:
            nao_value = self.nao_ds_split.sel(time=target_date).values.item()
        except KeyError:
            raise KeyError(
                f"No NAO index data available for date: {target_date}"
            )

        target = 1.0 if nao_value > 0 else 0.0
        return torch.tensor(target, dtype=torch.float32)

    def get_sample_slice(self, idx: int) -> slice:
        return super().get_sample_slice(idx)
