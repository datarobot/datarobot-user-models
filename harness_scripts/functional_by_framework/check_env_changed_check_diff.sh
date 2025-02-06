#!/usr/bin/env bash

# echo "configuring git"
# cat <<EOF > ${HOME}/.netrc
# machine ${DRONE_NETRC_MACHINE}
# login ${DRONE_NETRC_USERNAME}
# password ${DRONE_NETRC_PASSWORD}
# EOF
# echo "git configured"

[ ! -n $IMAGE ] && echo "IMAGE is not set" && exit 1
[ ! -n $CODEBASE_BRANCH ] && echo "CODEBASE_BRANCH is not set" && exit 1
[ ! -n $ENV_FOLDER ] && echo "ENV_FOLDER is not set" && exit 1
[ ! -n $FRAMEWORK ] && echo "FRAMEWORK is not set" && exit 1

echo "Step is running on image: $IMAGE"

git fetch origin master ${CODEBASE_BRANCH}
git branch -a

changed_paths=$(git diff --name-only origin/master...HEAD)
echo "changed_paths: $changed_paths"

test_image_tag_base=${ENV_FOLDER}_${FRAMEWORK}
if echo "${changed_paths}" | grep "${ENV_FOLDER}/${FRAMEWORK}" > /dev/null; then
    changed_deps=true;
    if [ -n $TRIGGER_PR_NUMBER ]; then
        test_image_tag=${test_image_tag_base}_${TRIGGER_PR_NUMBER};
    else
        test_image_tag=${test_image_tag_base}_${CODEBASE_BRANCH//\//_}
    fi
else
    changed_deps=false;
    test_image_tag=${test_image_tag_base}_latest;
fi

# Required by the Harness step
export changed_deps
export test_image_tag

echo "changed_deps: $changed_deps, test_image_tag: $test_image_tag"
