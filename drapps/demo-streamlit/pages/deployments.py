#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import os
import streamlit as st
from datarobot import Deployment
from urllib.parse import urlparse


def _make_deployment_link(parsed_url, deployment):
    deployment_url = parsed_url._replace(path=f"deployments/{deployment.id}").geturl()
    return f"[{deployment.label}]({deployment_url})"


@st.cache
def get_deployments_link_list():
    parsed_api_url = urlparse(os.getenv("DATAROBOT_ENDPOINT"))
    deployments = Deployment.list()
    return [_make_deployment_link(parsed_api_url, deployment) for deployment in deployments]


if __name__ == "__main__":
    st.header("Your model deployments")
    deployments_links = get_deployments_link_list()
    md_list = "\n".join([f"* {link}" for link in deployments_links])
    st.markdown(md_list)
