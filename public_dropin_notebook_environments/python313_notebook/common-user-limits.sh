#!/bin/bash

# Sets the max user processes (ulimit -u) via NOTEBOOKS_NPROC_LIMIT.
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

# Deliberately in the image layer: the nbx-operator chart shadows it with a read-only
# ConfigMap when notebookSession.userLimits is enabled, and relies on this write failing so
# the chart's version wins (CFX-6369). The write also fails at any non-build-time uid, where
# nothing sets the limits at all - so say which case it is instead of failing silently.
NBX_LIMITS_FILE=/etc/profile.d/bash-profile-load.sh
echo "Generating common bash profile..."
# One simple command with one redirection: bash does not propagate a failed redirection on a
# compound command through `!`, so `if ! { ...; } > file` would never fire.
if ! printf '%s\n' \
    "#!/bin/bash" \
    "# Setting user process limits." \
    "ulimit -Su ${nproc_limit}" \
    "ulimit -Hu ${nproc_limit}" > "$NBX_LIMITS_FILE" 2>/dev/null; then
    if [ -s "$NBX_LIMITS_FILE" ]; then
        echo "${NBX_LIMITS_FILE} is not writable and already has content - leaving it to the overlay that provides it." >&2
    else
        echo "WARNING: could not write ${NBX_LIMITS_FILE} and it is empty, so no process limits will be applied." >&2
        echo "WARNING: enable notebookSession.userLimits in the nbx-operator chart to supply it." >&2
    fi
fi
