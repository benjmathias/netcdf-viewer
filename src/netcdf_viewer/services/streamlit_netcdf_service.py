import logging

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from netcdf_viewer.viewers.streamlit_viewer import NetCDFStreamlitApp

logger = logging.getLogger(__name__)


class StreamlitNetCDFService:
    """Service class to handle the Streamlit NetCDF viewer application flow."""

    def __init__(self) -> None:
        """Initialize the Streamlit service."""
        self._streamlit = st
        self._app: NetCDFStreamlitApp = NetCDFStreamlitApp()
        self._app.streamlit = self._streamlit

    @property
    def streamlit(self):
        """Get the Streamlit instance.

        Returns:
            The Streamlit instance being used by the service.
        """
        return self._streamlit

    @property
    def app(self):
        """Get the NetCDFStreamlitApp instance.

        Returns:
            The NetCDFStreamlitApp instance being used by the service.
        """
        return self._app

    def run(self) -> None:
        """Run the Streamlit NetCDF viewer application."""
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
                file_info, data = self.app.parser.handle_uploaded_file(
                    uploaded_file_data
                )
                self.app.display_file_info(file_info, data)
                self.app.display_variables(data)
            except Exception as e:
                self.streamlit.error(f"Error processing file: {str(e)}")
