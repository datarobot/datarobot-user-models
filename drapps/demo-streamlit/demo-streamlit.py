#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import streamlit as st
from datarobot import Client
from datarobot.client import set_client


def start_streamlit():
    # Setup DR client
    # Because DATAROBOT_API_TOKEN and DATAROBOT_ENDPOINT environment variables provided
    # automatically, there is no need in manual setup
    set_client(Client())

    st.title("Custom Application streamlit demo")
    st.header("You can get list of your projects or model deployments")


if __name__ == "__main__":
    start_streamlit()
