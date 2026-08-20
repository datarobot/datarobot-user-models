# Copyright 2026 DataRobot, Inc. and its affiliates.
# All rights reserved.
# DataRobot, Inc. Confidential.
# This is unpublished proprietary source code of DataRobot, Inc.
# and its affiliates.
# The copyright notice above does not evidence any actual or intended
# publication of such source code.
#
# Serving-only execution environment for MCP servers. A deliberate narrowing of
# python311_genai_agents: same base image and uv/lock toolchain, only the
# `datarobot-genai[drmcp]` dependency line, and none of the Codespaces/Notebooks
# runtime (no sshd, no Jupyter kernel gateway, no IPython extensions, no
# monitoring agent, no DataRobot CLI). See DESIGN.md.
#
# Two deployment surfaces, and each constrains this file:
#
#   1. Custom model -- the platform runs /opt/code/start_server.sh, which creates
#      the venv at ${VENV_DIR}, `uv sync`es the user's project into it, and execs
#      `python -m app.main`.
#   2. Workload API (code-to-workload) -- this image is the BASE for a generated
#      Dockerfile that sets its own entrypoint and readiness probe.
#      start_server.sh is never invoked. That path is why the ENTRYPOINT is
#      explicitly CLEARED and no CMD is set (an inherited entrypoint is prefixed
#      to the generated one and fails immediately -- see the note at the bottom
#      of this file), and why ${VENV_PATH}/bin is
#      on PATH at image level so a bare `python`/`fastmcp` entrypoint resolves
#      without sourcing an activate script.

ARG WORKDIR=/etc/system/mcp
ARG VENV_PATH=${WORKDIR}/.venv
ARG UNAME=mcp
ARG UID=10101
ARG GID=10101
# Writable at build and runtime (custom model / uv); override with --build-arg if needed.
ARG UV_CACHE_DIR=/tmp/uv-cache

# Must stay the `-dev` variant rather than the slim runtime: start_server.sh runs
# `uv sync` at container start, so uv has to be present in the shipped image.
FROM datarobot/mirror_chainguard_datarobot.com_python-fips:3.12-dev AS base

ARG UNAME
ARG UID
ARG GID
ARG WORKDIR
ARG VENV_PATH
ARG UV_CACHE_DIR

USER root

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# git is here because a user project's `uv sync` may resolve a git dependency.
# hadolint ignore=DL3018,DL3059
RUN apk add --no-cache uv git curl

# Assert the uv version floor. Wolfi carries only the newest uv, so the `apk add` above
# cannot be pinned and the version floats with the apk index. pyproject.toml declares
# exclude-dependencies, which needs uv >= 0.10.0. An older uv rejects that unknown key,
# which fails parsing of the entire [tool.uv] table and takes its own required-version
# guard down with it. uv then only warns, exits 0, and resolves with the exclusions
# silently absent, quietly restoring the packages we dropped for CVE reasons. Fail the
# build here instead. Compared with bash builtins so this needs no awk or sort -V.
RUN v="$(uv --version)" && v="${v#uv }" && v="${v%% *}" \
    && IFS=. read -r maj min _ <<< "$v" \
    && if [ "$maj" -eq 0 ] && [ "$min" -lt 10 ]; then \
         echo "ERROR: uv $v is below the 0.10.0 floor required by exclude-dependencies" >&2; \
         exit 1; \
       fi \
    && echo "uv $v satisfies the >=0.10.0 exclude-dependencies floor"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX=/tmp/.python_cache \
    UV_COMPILE_BYTECODE=1 \
    VENV_PATH=${VENV_PATH} \
    PIP_NO_CACHE_DIR=1

ENV PATH="$VENV_PATH/bin:$PATH"

RUN uv venv ${VENV_PATH}
WORKDIR ${WORKDIR}

# Custom user to run the image from
RUN addgroup -g "$GID" "$UNAME" && \
    adduser -D -u "$UID" -G "$UNAME" -s /bin/bash "$UNAME"

RUN chown -R "$UNAME:$UNAME" ${WORKDIR} ${VENV_PATH} \
  && mkdir -p "${UV_CACHE_DIR}" \
  && chown -R "$UNAME:$UNAME" "${UV_CACHE_DIR}"

# Set after root `uv venv` / chown so Kaniko/BuildKit: `uv sync` as $UNAME does not inherit a
# root-owned ${UV_CACHE_DIR} (see sdists-v9/.git permission denied).
ENV UV_CACHE_DIR=${UV_CACHE_DIR}

USER $UNAME

FROM base AS builder

ENV UV_COMPILE_BYTECODE=1

ARG WORKDIR
ARG VENV_PATH
ARG UNAME
ARG UID
ARG GID
ARG UV_CACHE_DIR
ARG GIT_COMMIT

LABEL com.datarobot.repo-name="datarobot-user-models"
LABEL com.datarobot.repo-sha=$GIT_COMMIT

COPY --chown=${UID}:${GID} ./pyproject.toml ${WORKDIR}/pyproject.toml
COPY --chown=${UID}:${GID} ./uv.lock ${WORKDIR}/uv.lock

WORKDIR ${WORKDIR}
USER ${UNAME}

# Bakes the dependency set into ${VENV_PATH} (which Workload API entrypoints resolve
# through PATH) and, just as importantly, warms ${UV_CACHE_DIR} so the custom-model
# `uv sync` at container start resolves from cache instead of the network.
RUN . ${VENV_PATH}/bin/activate && \
    uv sync --frozen && \
    chmod -R 777 ${VENV_PATH}

USER root
# Custom model deployments and workloads run as securityContext.runAsUser: 1000, which is
# neither the build user nor root -- grant both the runtime venv target and the uv cache to
# all users to avoid access errors.
RUN mkdir -p /opt/venv /opt/code && chmod a+rwx /opt/venv
RUN chmod -R 777 "${UV_CACHE_DIR}"

# This is required for custom models to work with this image
COPY --chown=${UID}:${GID} ./start_server_custom_model.sh /opt/code/start_server.sh

# Last USER must be non-root. Platform runtimes still override this with
# securityContext.runAsUser: 1000; this is the default for docker run and scanners.
USER ${UNAME}

WORKDIR /
# PATH is deliberately not restated here -- `PATH="$VENV_PATH/bin:$PATH"` is set in the
# base stage and inherited, and re-prepending it would only duplicate the entry.
ENV VENV_DIR=/opt/venv \
    UV_CACHE_DIR=${UV_CACHE_DIR} \
    UV_LINK_MODE=copy \
    HOME=/opt \
    CODE_DIR=/opt/code \
    ADDRESS=0.0.0.0:8080 \
    PYTHONUNBUFFERED=1

# MCP server port. The only port this environment exposes.
EXPOSE 8080

# Clear the ENTRYPOINT, don't merely omit one. The chainguard python-fips base sets
# ENTRYPOINT ["/usr/bin/python"], so omitting the instruction INHERITS it -- and an
# inherited entrypoint is prefixed to whatever the Workload API's generated Dockerfile
# supplies, producing `python /usr/bin/python -m app.main` and an immediate failure.
# Both other MCP images in the fleet do the same for the same reason:
# ../dr_mcp_execute_sandbox_minimal/Dockerfile resets it to the venv interpreter, and
# af-component-datarobot-mcp's template Dockerfile carries a bare `ENTRYPOINT []`.
# python311_genai_agents does NOT do this; that path is untested because the MCP
# template defaults to a provided Dockerfile rather than a generated one.
ENTRYPOINT []

# No CMD either -- see the header note.
