import logging
import os
import tempfile
from datetime import datetime
from typing import Any

import netCDF4 as nc
import numpy as np

logger = logging.getLogger(__name__)


class NetCDFParser:
    """A generic class to handle NetCDF file parsing.

    This class provides methods to read, parse, and retrieve NetCDF file data.
    It handles large datasets by implementing optional downsampling and provides
    various data access methods.

    Attributes:
        coord_names (dict): Possible names for coordinate variables
    """

    coord_names: dict = {
        "latitude": ["lat", "latitude", "LATITUDE", "y"],
        "longitude": ["lon", "longitude", "LONGITUDE", "x"],
        "time": ["time", "TIME", "timestamp"],
    }

    def __init__(self) -> None:
        """Initialize the NetCDFParser."""
        self.dataset: nc.Dataset | None = None
        self.file_path: str | None = None

    def get_metadata_and_data_from_dataset(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read the already opened NetCDF dataset and extract its metadata.

        Returns:
            tuple: (file_info, data) containing file metadata and variable information

        Raises:
            RuntimeError: If the dataset is not opened
        """
        if self.dataset is None:
            raise RuntimeError("Dataset is not opened.")

        logger.debug(f"Reading NetCDF dataset: {self.dataset.filepath()}")

        file_info: dict[str, Any] = {
            "path": self.dataset.filepath(),
            "size": f"{os.path.getsize(self.dataset.filepath()) / (1024 * 1024):.2f} MB",
            "last_modified": datetime.fromtimestamp(
                os.path.getmtime(self.dataset.filepath())
            ),
            "dimensions": list(self.dataset.dimensions.keys()),
            "dim_details": {
                dim_name: {"size": str(len(dim)), "unlimited": dim.isunlimited()}
                for dim_name, dim in self.dataset.dimensions.items()
            },
            "global_attributes": {
                attr: str(getattr(self.dataset, attr)) 
                for attr in self.dataset.ncattrs()
            },
        }

        data: dict[str, Any] = {}
        for var_name, var in self.dataset.variables.items():
            var_info: dict[str, Any] = {
                "attributes": {
                    attr: str(getattr(var, attr)) 
                    for attr in var.ncattrs()
                },
                "shape": var.shape,
                "dtype": str(var.dtype),
                "dimensions": var.dimensions,
                "variable_name": var_name,
            }

            try:
                values = var[:]
                if isinstance(values, np.ma.MaskedArray):
                    logger.debug(
                        f"Variable '{var_name}' is a masked array. Converting to filled array with NaNs."
                    )
                    values = values.filled(np.nan)

                # Convert non-numeric data to strings
                if not np.issubdtype(values.dtype, np.number):
                    values = values.astype(str)
                    # Replace empty strings with 'N/A'
                    values[values == ''] = 'N/A'
                    values[values == ' '] = 'N/A'

                var_info["values"] = values

            except Exception as e:
                logger.error(f"Error retrieving data for variable '{var_name}': {e}")
                var_info["values"] = None

            data[var_name] = var_info

        logger.debug("NetCDF file parsed successfully.")
        return file_info, data

    def handle_uploaded_file(
        self, file_data: bytes
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Process the uploaded NetCDF file.

        Args:
            file_data: File data of the netcdf file

        Returns:
            tuple: (file_info, data) containing file metadata and variable information
        """
        tmp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp_file:
                tmp_file.write(file_data)
                tmp_file_path = tmp_file.name
                logger.debug(f"Temporary file created at {tmp_file_path}")

            # Open the dataset and assign to the class attribute
            self.dataset = nc.Dataset(tmp_file_path, "r")
            self.file_path = tmp_file_path
            logger.debug(f"NetCDF dataset {tmp_file_path} is now open.")

            # Read the dataset
            file_info, data = self.get_metadata_and_data_from_dataset()
            return file_info, data
        except Exception as e:
            logger.error(f"Failed to read NetCDF file: {e}")
            raise e
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                if self.dataset is not None:
                    try:
                        self.dataset.close()
                        logger.debug("NetCDF dataset closed.")
                    except Exception as close_exc:
                        logger.warning(f"Failed to close dataset: {close_exc}")
                os.remove(tmp_file_path)
                logger.debug(f"Temporary file {tmp_file_path} deleted.")
                self.dataset = None  # Ensure dataset reference is cleared
                self.file_path = None

    def get_variable_data(
        self, var_data: dict[str, Any]
    ) -> np.ndarray:
        """Get variable data without downsampling.

        Args:
            var_data: Variable metadata

        Returns:
            np.ndarray: Variable data
        """
        if self.dataset is None:
            logger.error("Dataset is not opened.")
            raise RuntimeError("Dataset is not opened.")

        var_obj: nc.Variable = self.dataset.variables[var_data["variable_name"]]
        values: np.ndarray = var_obj[:]

        # Convert masked arrays to regular arrays with NaNs
        if isinstance(values, np.ma.MaskedArray):
            logger.debug(
                f"Variable '{var_data['variable_name']}' is a masked array. Converting to filled array with NaNs."
            )
            values = values.filled(np.nan)

        return values
