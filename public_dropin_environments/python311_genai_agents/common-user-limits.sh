#!/bin/bash

# Per-UID `ulimit -u` (max user processes) is shared across pods on a node;
# override via NOTEBOOKS_NPROC_LIMIT.
# Coerce to a positive integer; fall back to default if the env value is
# missing, non-numeric, or zero/negative (don't trust the value blindly: it is
# interpolated into a script that gets sourced from /etc/profile.d).
DEFAULT_NPROC_LIMIT=8192
if [ -z "${NOTEBOOKS_NPROC_LIMIT:-}" ]; then
    echo "NOTEBOOKS_NPROC_LIMIT not set, defaulting to ${DEFAULT_NPROC_LIMIT}." >&2
    nproc_limit=$DEFAULT_NPROC_LIMIT
elif ! [[ "$NOTEBOOKS_NPROC_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
    echo "NOTEBOOKS_NPROC_LIMIT='${NOTEBOOKS_NPROC_LIMIT}' is not a positive integer, defaulting to ${DEFAULT_NPROC_LIMIT}." >&2
    nproc_limit=$DEFAULT_NPROC_LIMIT
else
    nproc_limit=$NOTEBOOKS_NPROC_LIMIT
fi

echo "Generating common bash profile..."
{
    echo "#!/bin/bash"
    echo "# Setting user process limits."
    echo "ulimit -Su ${nproc_limit}"
    echo "ulimit -Hu ${nproc_limit}"
} > /etc/profile.d/bash-profile-load.sh
