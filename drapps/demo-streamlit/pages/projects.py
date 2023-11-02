#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import os
import streamlit as st
from datarobot import Project
from urllib.parse import urlparse


def _make_project_link(parsed_url, project):
    project_url = parsed_url._replace(path=f"projects/{project.id}").geturl()
    return f"[{project.project_name}]({project_url})"


@st.cache
def get_projects_link_list():
    parsed_api_url = urlparse(os.getenv("DATAROBOT_ENDPOINT"))
    projects = Project.list()
    return [_make_project_link(parsed_api_url, project) for project in projects]


if __name__ == "__main__":
    st.header("Your projects")
    projects_links = get_projects_link_list()
    md_list = "\n".join([f"* {link}" for link in projects_links])
    st.markdown(md_list)
