PYTHON_VERSION = 3.12
ENV_VERSIONID  = $(shell jq -r .environmentVersionId env_info.json)
ENV_REPONAME   = $(shell jq -r .imageRepository env_info.json)
LABEL_REPONAME = $(notdir $(shell git remote get-url origin || printf 'unknown'))
LABEL_REPOSHA  = $(shell git rev-parse HEAD 2>/dev/null || printf 'unknown')

update-deps:
	docker run -v "$(PWD):/work" --entrypoint /bin/bash "datarobotdev/py-gen-requirements:$(PYTHON_VERSION)" \
	  -c 'uv pip compile --no-annotate --no-emit-index-url requirements.in -o requirements.txt'

image:
	docker build . \
	  --build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
	  -t $(ENV_REPONAME):$(ENV_VERSIONID) \
	  -t datarobotdev/$(ENV_REPONAME):$(ENV_VERSIONID)

image-push:
	docker push datarobotdev/$(ENV_REPONAME):$(ENV_VERSIONID)
