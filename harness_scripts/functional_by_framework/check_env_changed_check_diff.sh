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

# by default set namespace, repo, and tag as
# datarobot/<image_repository>:<env_version_id>
# e.g. datarobot/env-python-sklearn:12355123abc918234
env_info="${ENV_FOLDER}/${FRAMEWORK}/env_info.json"
ENV_VERSION_ID=$(jq -r '.environmentVersionId' ${env_info})

# once we implement image promotion, change it to datarobot.
test_image_namespace=datarobotdev
test_image_tag=${ENV_VERSION_ID}

changed_deps=false;

IMAGE_REPOSITORY=$(jq -r '.imageRepository' ${env_info})
if [ "${IMAGE_REPOSITORY}" = "null" ]; then
  echo "Image repository must be defined in 'imageRepository' in env_info.json"
else
  # e.g. datarobot/env-python-sklearn:12355123abc918234
  echo "Image repo read from env_info.json: ${IMAGE_REPOSITORY}"
  test_image_repository=${IMAGE_REPOSITORY}
fi


if echo "${changed_paths}" | grep "${ENV_FOLDER}/${FRAMEWORK}" > /dev/null; then
    changed_deps=true;
    # if env changed, means we want to push tmp image into datarobotdev
    test_image_namespace=datarobotdev
    if [ -n $TRIGGER_PR_NUMBER ] && [ "$TRIGGER_PR_NUMBER" != "null" ]; then
        # datarobotdev/env-python-sklearn:12355123abc918234_PR_NUM
        # placeholder in case we want to add PR number back,
        # but then it will be difficult to promote
        test_image_tag=${test_image_tag}
    else
        test_image_tag=${test_image_tag}_${CODEBASE_BRANCH}
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
