REPO       = dropin-env-base
PATCH     := 1
TAG        = py311-$(PATCH)
BASE_IMAGE = python:3.11.11-slim-bookworm

#debian11:py3.9-jre11.0.16-drum1.16.12-mlops10.2.8


all:
	docker build --no-cache \
	  --build-arg BASE_ROOT_IMAGE=$(BASE_IMAGE) \
	  -t datarobot/$(REPO):$(TAG) -t datarobotdev/$(REPO):$(TAG) .
#:
#IMAGE_NAME_DATAROBOT=${IMAGE_ORG_DATAROBOT}/${IMAGE_REPO}:${IMAGE_TAG}
#IMAGE_NAME_DATAROBOTDEV=${IMAGE_ORG_DATAROBOTDEV}/${IMAGE_REPO}:${IMAGE_TAG}
