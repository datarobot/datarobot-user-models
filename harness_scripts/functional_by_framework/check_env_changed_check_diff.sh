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
echo "changed_paths: $changed_paths"

test_image_repository=datarobot-user-models
test_image_tag_base=${ENV_FOLDER}_${FRAMEWORK}

yaml_file="${ENV_FOLDER}/${FRAMEWORK}/values.yaml"
if [ -f "${yaml_file}" ]; then
    ls -la ${yaml_file}
    ENV_VERSION_ID=$(yq '.environmentVersionId' ${yaml_file})
    IMAGE_REPOSITORY=$(yq '.imageRepository' ${yaml_file})
    echo ${ENV_VERSION_ID}
    echo ${IMAGE_REPOSITORY}
    test_image_repository=${IMAGE_REPOSITORY}
    test_image_tag_base=${ENV_VERSION_ID}
else
    echo "Yaml file does not exist, continue with old `datarobot-user-models` images"
fi

if echo "${changed_paths}" | grep "${ENV_FOLDER}/${FRAMEWORK}" > /dev/null; then
    changed_deps=true;
    test_image_namespace=datarobotdev
    if [ -n $TRIGGER_PR_NUMBER ] && [ "$TRIGGER_PR_NUMBER" != "null" ]; then
        test_image_tag=${test_image_tag_base}_${TRIGGER_PR_NUMBER};
    else
        test_image_tag=${test_image_tag_base}_${CODEBASE_BRANCH}
        # If the test_image_tag may contain a slash, replace it with an underscore (POSIX compliant)
        while case $test_image_tag in */*) true;; *) false;; esac; do
          test_image_tag=${test_image_tag%%/*}_${test_image_tag#*/}
        done
    fi
else
    changed_deps=false;
    # should be changed when switch to yaml
    test_image_namespace=datarobotdev
    test_image_tag=${test_image_tag_base}_latest;

fi

# If the environment variable 'USE_LOCAL_DOCKERFILE' is set to "true", than add '.local'
#if [ -n $USE_LOCAL_DOCKERFILE ] && [ "$USE_LOCAL_DOCKERFILE" = "true" ]; then
#    test_image_tag=${test_image_tag}.local
#fi

# Required by the Harness step
export changed_deps
export test_image_namespace
export test_image_repository
export test_image_tag

echo "changed_deps: $changed_deps"
echo "test_image_namespace: $test_image_namespace"
echo "test_image_repository: $test_image_repository"
echo "test_image_tag: $test_image_tag"
