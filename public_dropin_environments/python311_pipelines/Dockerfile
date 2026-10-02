# Base image for DataRobot Pipelines images on Python 3.11.
# pipelines-api layers the user's packages and its runner scripts on top of this
# image at build time; nothing here runs on its own (no ENTRYPOINT).
#
# This is a private chain-guard development image that is stored in DataRobot's private registry.
# Replace it with your own development chain-guard image if you build your own.
ARG PYTHON_VERSION=3.11
ARG INSTALLROOT=/var/lib/covalent

FROM datarobot/mirror_chainguard_datarobot.com_python-fips:${PYTHON_VERSION}-dev AS build
ARG INSTALLROOT
USER root
WORKDIR /build
COPY requirements.txt .
# The venv lives at its final path so the interpreter and scripts carry no /build prefix.
RUN apk update && apk add --no-cache bash build-base git && \
    python -m venv --copies ${INSTALLROOT} && \
    ${INSTALLROOT}/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    ${INSTALLROOT}/bin/pip install --no-cache-dir -r requirements.txt
# cloudpickle source clone: at task start the runner checks out the version the
# client pickled with (handle_cloudpickle.py in pipelines-api), so the clone must
# exist and be writable in the final image.
RUN cd ${INSTALLROOT}/bin && git clone https://github.com/cloudpipe/cloudpickle.git
# Third-party pins for the user-package layer (`uv pip install -c constraints.txt`).
RUN ${INSTALLROOT}/bin/pip freeze --exclude-editable \
    | grep -viE '^(datarobot|covalent)' > ${INSTALLROOT}/constraints.txt
# Shared libraries the production base lacks: git's, OpenMP for numeric wheels,
# and libcrypt for /bin/sh (dropped from the 3.13+ production base).
RUN mkdir -p /tmp/git-libs /tmp/openmp-libs /tmp/sh-libs && \
    cp -L /usr/lib/libpcre2-8.so.0 /usr/lib/libz.so.1 /usr/lib/libcurl.so.4 /usr/lib/libzstd.so.1 /tmp/git-libs/ && \
    cp -L /usr/lib/libgomp.so.1 /tmp/openmp-libs/ && \
    cp -L /usr/lib/libcrypt.so.1 /tmp/sh-libs/

# This is a private production chain-guard image that is stored in DataRobot's private registry.
FROM datarobot/mirror_chainguard_datarobot.com_python-fips:${PYTHON_VERSION}
ARG INSTALLROOT
ENV PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=${INSTALLROOT}/lib \
    INSTALLROOT=${INSTALLROOT} \
    COVALENT_ROOT=${INSTALLROOT} \
    COVALENT_CONFIG_DIR=/etc/covalent \
    COVALENT_PLUGINS_DIR=/etc/covalent/plugins \
    COVALENT_LOGDIR=/var/log/covalent \
    COVALENT_CACHE_DIR=/var/cache/covalent \
    COVALENT_ENABLE_LIVE_STDOUT=1 \
    COVALENT_DISPATCH_BASE_ENDPOINT=/api/v2/job/lattices \
    COVALENT_HOME=/home/covalent \
    HOME=/home/covalent
LABEL org.opencontainers.image.title="[DataRobot] Python 3.11 Pipelines execution environment"
LABEL org.opencontainers.image.vendor="DataRobot"
LABEL org.opencontainers.image.source="https://github.com/datarobot/datarobot-user-models"
LABEL org.opencontainers.image.documentation="https://github.com/datarobot/datarobot-user-models/tree/master/public_dropin_environments/python311_pipelines"
USER root
WORKDIR ${INSTALLROOT}
COPY --from=build ${INSTALLROOT}/ .
# busybox + sh: the runner's entry script and the user-package layer need a shell.
COPY --from=build /bin/busybox /bin/busybox
COPY --from=build /bin/sh /bin/sh
COPY --from=build /tmp/sh-libs/* /usr/lib/
RUN /bin/busybox --install -s /bin
COPY --from=build /usr/bin/git /usr/bin/git
COPY --from=build /usr/libexec/git-core /usr/libexec/git-core
COPY --from=build /tmp/git-libs/* /usr/lib/
COPY --from=build /tmp/openmp-libs/* /usr/lib/
RUN chmod -R 755 ${INSTALLROOT} && \
    chmod -R 777 ${INSTALLROOT}/bin/cloudpickle && \
    chmod -R 777 ${INSTALLROOT}/lib/python3.*/site-packages/cloudpickle && \
    /usr/bin/git config --system --add safe.directory ${INSTALLROOT}/bin/cloudpickle
# Expose NVIDIA wheel libraries when a user installs CUDA packages on top.
RUN for so in ${INSTALLROOT}/lib/python*/site-packages/nvidia/*/lib/*.so*; do \
        [ -e "$so" ] && ln -sf "$so" ${INSTALLROOT}/lib/ ; \
    done; true
RUN mkdir -p /etc/covalent /etc/covalent/plugins /var/log/covalent /var/cache/covalent \
        /scratch /volumes /home/covalent /opt/covalent /opt/pipelines-env && \
    chmod 1777 /etc/covalent /etc/covalent/plugins /var/log/covalent /var/cache/covalent \
        /scratch /volumes /home/covalent && \
    chmod 755 /opt/covalent /opt/pipelines-env
COPY --chmod=0755 smoke_test.py /opt/pipelines-env/smoke_test.py
ENV PATH=${INSTALLROOT}/bin:${PATH}
RUN touch /etc/environment && \
    (grep -v '^PATH=' /etc/environment > /tmp/environment || true) && \
    echo "PATH=${INSTALLROOT}/bin:${PATH}" >> /tmp/environment && \
    mv /tmp/environment /etc/environment
# Slim: build-only artefacts. pip stays out; the user-package layer uses uv.
RUN rm -rf ${INSTALLROOT}/include ${INSTALLROOT}/lib/python3.*/site-packages/pip* \
        ${INSTALLROOT}/lib/python3.*/idlelib ${INSTALLROOT}/lib/python3.*/ensurepip && \
    find ${INSTALLROOT} -name '__pycache__' -type d -exec rm -rf '{}' '+' || true && \
    find ${INSTALLROOT} -name '*.a' -type f -delete || true
WORKDIR /home/covalent
USER 65532
# Contract check as the runtime user: a dependency bump that breaks what
# pipelines-api layers onto this image fails the image build here.
RUN python /opt/pipelines-env/smoke_test.py
