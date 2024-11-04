import logging

from netcdf_viewer.services.streamlit_netcdf_service import (
    StreamlitNetCDFService,
)

# Logging configuration
logging.basicConfig(level=logging.INFO)


def main():
    """Main function to run the Streamlit application."""

    service = StreamlitNetCDFService()
    service.run()
