PYTHON_VERSION = 3.11
ENV_VERSIONID  = $(shell jq -r .environmentVersionId env_info.json)
ENV_REPONAME   = $(shell jq -r .imageRepository env_info.json)
LABEL_REPONAME = $(notdir $(shell git remote get-url origin || printf 'unknown'))
LABEL_REPOSHA  = $(shell git rev-parse HEAD 2>/dev/null || printf 'unknown')

update-deps:
	$(info requirements.txt is manually managed for this environment)

image:
	docker build . \
	  --build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
	  --build-arg GIT_COMMIT=$(shell git rev-parse HEAD) \
	  --label=com.datarobot.repo-sha=$(LABEL_REPOSHA) \
	  --label=com.datarobot.repo-name=$(LABEL_REPONAME) \
	  -t $(ENV_REPONAME):$(ENV_VERSIONID) \
	  -t datarobotdev/$(ENV_REPONAME):$(ENV_VERSIONID)

image-push:
	docker push datarobotdev/$(ENV_REPONAME):$(ENV_VERSIONID)
