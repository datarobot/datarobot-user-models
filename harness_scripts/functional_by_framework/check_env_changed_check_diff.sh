#!/bin/sh

# echo "configuring git"
# cat <<EOF > ${HOME}/.netrc
# machine ${DRONE_NETRC_MACHINE}
# login ${DRONE_NETRC_USERNAME}
# password ${DRONE_NETRC_PASSWORD}
# EOF
# echo "git configured"

[ ! -n $CODEBASE_BRANCH ] && echo "CODEBASE_BRANCH is not set" && exit 1
[ ! -n $ENV_FOLDER ] && echo "ENV_FOLDER is not set" && exit 1
[ ! -n $FRAMEWORK ] && echo "FRAMEWORK is not set" && exit 1

git fetch origin master ${CODEBASE_BRANCH}
git branch -a

changed_paths=$(git diff --name-only origin/master...HEAD)
echo "--- changed paths ---"
echo "${changed_paths}"
echo "--- --- --- --- ---"

# by default define namespace, repo, and tag for the existing flow without changes
# e.g. datarobotdev/datarobot-user-models:public_dropin_envs_python3_sklearn_latest
test_image_namespace=datarobotdev
test_image_repository=datarobot-user-models

test_image_tag_base=${ENV_FOLDER}_${FRAMEWORK}
test_image_tag=${test_image_tag_base}_latest;

changed_deps=false;
env_info="${ENV_FOLDER}/${FRAMEWORK}/env_info.json"

IMAGE_REPOSITORY=$(jq -r '.imageRepository' ${env_info})
if [ "${IMAGE_REPOSITORY}" = "null" ]; then
  echo "Image repository is not defined in env_info.json"
else
  # if env_info has imageRepository
  # point test_image_namespace to datarobot
  # point test_image_repository to defined repo
  # point tag to ENV_VERSION_ID
  # e.g. datarobot/env-python-sklearn:12355123abc918234
  ENV_VERSION_ID=$(jq -r '.environmentVersionId' ${env_info})
  echo "read: ${ENV_VERSION_ID}"
  echo "read ${IMAGE_REPOSITORY}"
  test_image_namespace=datarobot
  test_image_repository=${IMAGE_REPOSITORY}
  test_image_tag_base=${ENV_VERSION_ID}
  test_image_tag=${ENV_VERSION_ID}
fi


if echo "${changed_paths}" | grep "${ENV_FOLDER}/${FRAMEWORK}" > /dev/null; then
    changed_deps=true;
    # if env changed, means we want to push tmp image into datarobotdev
    test_image_namespace=datarobotdev
    if [ -n $TRIGGER_PR_NUMBER ] && [ "$TRIGGER_PR_NUMBER" != "null" ]; then
        # datarobotdev/env-python-sklearn:12355123abc918234_PR_NUM
        # or
        # datarobotdev/datarobot-user-models:public_dropin_envs_python3_sklearn_PR_NUM
        test_image_tag=${test_image_tag_base}_${TRIGGER_PR_NUMBER};
    else
        test_image_tag=${test_image_tag_base}_${CODEBASE_BRANCH}
        # If the test_image_tag may contain a slash, replace it with an underscore (POSIX compliant)
        while case $test_image_tag in */*) true;; *) false;; esac; do
          test_image_tag=${test_image_tag%%/*}_${test_image_tag#*/}
        done
    fi
fi

# Required by the Harness step
export changed_deps
export test_image_namespace
export test_image_repository
export test_image_tag

echo "changed_deps: $changed_deps"
echo "test_image_namespace: $test_image_namespace"
echo "test_image_repository: $test_image_repository"
echo "test_image_tag: $test_image_tag"
