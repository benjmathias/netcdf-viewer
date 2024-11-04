import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.graph_objs import Figure
from streamlit.runtime.uploaded_file_manager import UploadedFile

from netcdf_viewer.parsers.netcdf_parser import NetCDFParser

logger = logging.getLogger(__name__)

# Constants for plot dimensions
SCATTER_PLOT_WIDTH: int = 800
SCATTER_PLOT_HEIGHT: int = 600
DENSITY_PLOT_THRESHOLD: int = 50000    # Switch to density plot above this threshold


class NetCDFStreamlitApp:
    """Streamlit application for interacting with NetCDF files.

    This class handles the user interface, file uploads, and visualization
    by leveraging the NetCDFParser class.
    """

    def __init__(self) -> None:
        """Initialize the Streamlit application."""
        self.parser = NetCDFParser()
        self._streamlit = st

    @property
    def streamlit(self):
        """Get the Streamlit instance.

        Returns:
            The Streamlit instance being used by the application.
        """
        return self._streamlit

    @streamlit.setter
    def streamlit(self, value):
        """Set the Streamlit instance.

        Args:
            value: The Streamlit instance to use.
        """
        self._streamlit = value

    @st.cache_data
    def get_data(self, file_data: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
        """Cache the parsed data to improve performance."""
        parser = NetCDFParser()
        file_info, data = parser.handle_uploaded_file(file_data)
        return file_info, data

    def run(self) -> None:
        """Run the Streamlit application."""
        self.streamlit.set_page_config(
            page_title="NetCDF File Previewer", layout="wide"
        )

        self.streamlit.title("NetCDF File Previewer")

        uploaded_file: UploadedFile = self.streamlit.file_uploader(
            "Choose a NetCDF file", type=["nc"]
        )

        if uploaded_file is not None:
            try:
                uploaded_file_data: bytes = uploaded_file.read()
                file_info, data = self.get_data(uploaded_file_data)
                self.display_file_info(file_info, data)
                self.display_variables(data)
            except Exception as e:
                self.streamlit.error(f"Error processing file: {str(e)}")

    def display_file_info(
        self, file_info: dict[str, Any], data: dict[str, Any]
    ) -> None:
        """Display file information in the Streamlit interface.

        Args:
            file_info (dict): File metadata
            data (dict): Variable data
        """
        self.streamlit.header("File Information")
        col1, col2, col3 = self.streamlit.columns(3)
        with col1:
            self.streamlit.metric("File Size", file_info["size"])
        with col2:
            self.streamlit.metric("Number of Variables", len(data))
        with col3:
            self.streamlit.metric("Number of Dimensions", len(file_info["dimensions"]))

        # Dimensions
        self.streamlit.subheader("Dimensions")
        dim_df: pd.DataFrame = pd.DataFrame(
            [
                {
                    "Dimension": dim,
                    "Size": details["size"],
                    "Unlimited": "✓" if details["unlimited"] else "✗",
                }
                for dim, details in file_info["dim_details"].items()
            ]
        )
        self.streamlit.dataframe(dim_df, use_container_width=True, hide_index=True)

        # Global Attributes
        if file_info["global_attributes"]:
            self.streamlit.subheader("Global Attributes")
            attr_df: pd.DataFrame = pd.DataFrame(
                [
                    {"Attribute": attr, "Value": value}
                    for attr, value in file_info["global_attributes"].items()
                ]
            )
            self.streamlit.dataframe(attr_df, use_container_width=True, hide_index=True)

    def display_variables(self, data: dict[str, Any]) -> None:
        """Display variable information and visualizations."""
        self.streamlit.header("Variables")
        selected_var: str | None = self.streamlit.selectbox(
            "Select a variable to explore:", list(data.keys())
        )

        if selected_var:
            var_data: dict[str, Any] = data[selected_var]
            var_name: str = var_data["variable_name"]

            # Display metadata without loading full dataset
            size_mb: float = np.prod(var_data['shape']) * np.dtype(var_data['dtype']).itemsize / (1024 * 1024)
            self.streamlit.write(f"Variable size: {size_mb:.2f} MB")

            col1, col2 = self.streamlit.columns(2)
            with col1:
                self.streamlit.subheader("Variable Information")
                self.streamlit.write(f"**Shape:** {var_data['shape']}")
                self.streamlit.write(f"**Data Type:** {var_data['dtype']}")
                self.streamlit.write(f"**Dimensions:** {var_data['dimensions']}")

            with col2:
                self.streamlit.subheader("Variable Attributes")
                if var_data["attributes"]:
                    for attr, value in var_data["attributes"].items():
                        self.streamlit.write(f"**{attr}:** {value}")
                else:
                    self.streamlit.write("No attributes found")

            self.streamlit.subheader("Data Visualization")
            
            try:
                dims = var_data["dimensions"]
                x_dim_idx, y_dim_idx = self._get_dimension_selections(dims)
                
                if x_dim_idx != y_dim_idx:
                    # Add slider for scatter plot resolution just before plotting
                    max_points = self.streamlit.slider(
                        "Scatter Plot Resolution",
                        min_value=1_000,
                        max_value=49_000,
                        step=1_000,
                        value=15_000,
                    )

                    # Only load and process the data when needed
                    values_np = self._load_data_with_stride(var_data, max_points)
                    if values_np is not None:
                        values_transposed, num_nans = self._transpose_data(values_np, x_dim_idx, y_dim_idx)
                        
                        if values_transposed is not None:
                            fig: Figure = self._plot_variable(
                                var_data,
                                selected_var,
                                values_transposed,
                                max_points,
                                x_dim_idx,
                                y_dim_idx
                            )
                            self.streamlit.plotly_chart(fig, use_container_width=True)

                            if num_nans > 0:
                                self.streamlit.warning(f"Data contains {num_nans} NaN values which will not be plotted.")
                        
                            # Add raw data preview after the plot
                            self._display_raw_data_preview(values_transposed, var_data["dimensions"], x_dim_idx, y_dim_idx)
            except Exception as e:
                self.streamlit.error(f"Could not create plot: {str(e)}")

    def _load_data_with_stride(self, var_data: dict[str, Any], max_points: int) -> np.ndarray | None:
        """Load data with appropriate stride based on desired number of points."""
        total_points = np.prod(var_data['shape'])
        
        # Calculate stride to achieve approximately max_points
        stride = max(1, int(np.ceil(np.sqrt(total_points / max_points))))
        
        try:
            # Create slice objects with stride
            slices = tuple(slice(None, None, stride) for _ in var_data['shape'])
            
            # Get the data with stride
            values = var_data['values'][slices]
            
            if isinstance(values, np.ma.MaskedArray):
                values = values.filled(np.nan)
                
            return values
        except Exception as e:
            self.streamlit.error(f"Error loading data: {e}")
            return None

    def _plot_variable(
        self, var_data: dict[str, Any], var_name: str, values_transposed: np.ndarray, max_points: int, x_dim_idx: int, y_dim_idx: int
    ) -> Figure:
        """Create an appropriate plot for the variable."""
        if not self._is_plottable(var_data):
            return px.scatter()

        if values_transposed is None:
            return px.scatter()

        total_points = values_transposed.size
        if total_points > DENSITY_PLOT_THRESHOLD:
            return self._create_density_plot(values_transposed, var_data["dimensions"], x_dim_idx, y_dim_idx, var_name)
        else:
            return self._create_scatter_plot(values_transposed, var_data["dimensions"], x_dim_idx, y_dim_idx, var_name)

    def _create_density_plot(
        self, 
        values: np.ndarray, 
        dims: list[str], 
        x_dim_idx: int, 
        y_dim_idx: int, 
        var_name: str
    ) -> Figure:
        """Create a density heatmap for large datasets."""
        y, x = np.mgrid[0:values.shape[0], 0:values.shape[1]]
        
        fig = px.density_heatmap(
            x=x.flatten(),
            y=y.flatten(),
            z=values.flatten(),
            title=f"{var_name} Density Plot",
            labels={
                "x": dims[x_dim_idx],
                "y": dims[y_dim_idx],
                "z": var_name,
            },
            width=SCATTER_PLOT_WIDTH,
            height=SCATTER_PLOT_HEIGHT,
        )

        fig.update_layout(
            margin=dict(l=40, r=40, t=50, b=40),
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
                autorange="reversed",
            ),
        )

        return fig

    def _create_scatter_plot(
        self, 
        values: np.ndarray, 
        dims: list[str], 
        x_dim_idx: int, 
        y_dim_idx: int, 
        var_name: str,
    ) -> Figure:
        """Create the scatter plot with the processed data."""
        y, x = np.mgrid[0:values.shape[0], 0:values.shape[1]]
        
        # Flatten arrays
        x_flat = x.flatten()
        y_flat = y.flatten()
        values_flat = values.flatten()

        fig = px.scatter(
            x=x_flat,
            y=y_flat,
            color=values_flat,
            color_continuous_scale="Viridis",
            title=f"{var_name} Scatter Plot",
            labels={
                "x": dims[x_dim_idx],
                "y": dims[y_dim_idx],
                "color": var_name,
            },
            width=SCATTER_PLOT_WIDTH,
            height=SCATTER_PLOT_HEIGHT,
        )

        fig.update_layout(
            margin=dict(l=40, r=40, t=50, b=40),
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
                autorange="reversed",
            ),
        )

        fig.update_traces(
            marker=dict(size=3),
            showlegend=False,
        )

        return fig

    def _is_plottable(self, var_data: dict[str, Any]) -> bool:
        """Check if variable can be plotted."""
        if len(var_data["dimensions"]) < 2:
            self.streamlit.warning(
                "Selected variable is not 2D and cannot be plotted as a scatter plot."
            )
            return False
        return True

    def _get_dimension_selections(self, dims: list[str]) -> tuple[int, int]:
        """Get user selections for plot dimensions."""
        col_x, col_y = self.streamlit.columns(2)
        with col_x:
            x_dim_idx = self.streamlit.selectbox(
                "Select X-axis dimension:",
                options=range(len(dims)),
                format_func=lambda x: dims[x],
                key="x_dim_default",
                index=0,
            )
        with col_y:
            y_dim_idx = self.streamlit.selectbox(
                "Select Y-axis dimension:",
                options=range(len(dims)),
                format_func=lambda x: dims[x],
                key="y_dim_default",
                index=1,
            )
        return x_dim_idx, y_dim_idx

    def _transpose_data(
        self, values_np: np.ndarray, x_dim_idx: int, y_dim_idx: int
    ) -> tuple[np.ndarray | None, int]:
        """Transpose data for plotting and count NaN values."""
        try:
            values_transposed = np.moveaxis(values_np, [x_dim_idx, y_dim_idx], [-1, -2])
            
            if not np.issubdtype(values_transposed.dtype, np.number):
                self.streamlit.warning("Cannot plot non-numeric data.")
                return None, 0

            num_nans = np.isnan(values_transposed).sum()
            
            return values_transposed, num_nans
        except Exception as e:
            self.streamlit.error(f"Error processing data: {e}")
            return None, 0

    def _display_raw_data_preview(self, values: np.ndarray, dims: list[str], x_dim_idx: int, y_dim_idx: int) -> None:
        """Display a preview of the raw data as a matrix."""
        self.streamlit.subheader("Raw Data Preview")
        try:
            # Add controls in columns
            col1, col2 = self.streamlit.columns(2)
            with col1:
                hide_nan_rows = self.streamlit.checkbox("Hide rows with all NaN values", value=True)
                max_rows = self.streamlit.slider(
                    f"Max {dims[y_dim_idx]} rows",
                    min_value=10,
                    max_value=100,
                    value=50,
                )
            with col2:
                hide_nan_cols = self.streamlit.checkbox("Hide columns with all NaN values", value=True)
                max_cols = self.streamlit.slider(
                    f"Max {dims[x_dim_idx]} columns",
                    min_value=10,
                    max_value=100,
                    value=20,
                )

            preview = self._get_preview_data(values, max_rows, max_cols)
            df = self._create_preview_dataframe(preview, dims[x_dim_idx], dims[y_dim_idx])

            # Filter NaN rows and columns if requested
            if hide_nan_rows:
                df = df.dropna(how='all')
            if hide_nan_cols:
                df = df.dropna(axis=1, how='all')

            self.streamlit.dataframe(df, use_container_width=True)
        except Exception as e:
            self.streamlit.error(f"Could not display raw data preview: {str(e)}")

    def _get_preview_data(self, values: np.ndarray, max_rows: int = 50, max_cols: int = 20) -> np.ndarray:
        """Get a subset of the data for preview."""
        # Handle masked arrays
        if isinstance(values, np.ma.MaskedArray):
            values = values.filled(np.nan)
        
        # Take a slice of the data
        preview = values[:max_rows, :max_cols]
        return preview

    def _create_preview_dataframe(self, preview: np.ndarray, x_dim: str, y_dim: str) -> pd.DataFrame:
        """Create a DataFrame from the preview data with proper indexing."""
        df = pd.DataFrame(
            preview,
            columns=[f"{x_dim}_{i}" for i in range(preview.shape[1])],
            index=[f"{y_dim}_{i}" for i in range(preview.shape[0])]
        )
        return df
