# This is a private chain-guard development image that is stored in DataRobot's private registry.
# Replace it with your own development chain-gaurd image if you build your own.
ARG BASE_ROOT_IMAGE=datarobot/mirror_chainguard_datarobot.com_python-fips:3.11-dev
FROM ${BASE_ROOT_IMAGE} AS build

USER root

# Install R and required R packages using apk and Rscript
RUN apk add --no-cache \
    R \
    R-dev
#     openblas \
#     harfbuzz \
#     pkgconf \
#     fontconfig fontconfig-dev \
#     libsodium-dev \
#     openssl-dev \
#     libxml2-dev \
#     freetype-dev \
#     harfbuzz-dev \
#     fribidi-dev \
#     libpng-dev \
#     tiff-dev \
#     libjpeg-turbo-dev && \
#     ln -s /usr/lib/R/lib /usr/lib/R/lib64

# This is a private production chain-guard image that is stored in DataRobot's private registry.
# Replace it with your own production chain-gaurd image if you build your own.
FROM datarobot/mirror_chainguard_datarobot.com_python-fips:3.11

USER root

# Copy R libraries and binaries from the build stage
COPY --from=build /usr/lib/libpcre2-8* /usr/lib/
