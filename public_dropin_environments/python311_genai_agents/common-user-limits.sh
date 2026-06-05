#!/bin/bash

# Per-UID PID cap is shared across pods on a node; override via NOTEBOOKS_NPROC_LIMIT.
: "${NOTEBOOKS_NPROC_LIMIT:=8192}"

echo "Generating common bash profile..."
{
    echo "#!/bin/bash"
    echo "# Setting user process limits."
    echo "ulimit -Su ${NOTEBOOKS_NPROC_LIMIT}"
    echo "ulimit -Hu ${NOTEBOOKS_NPROC_LIMIT}"
} > /etc/profile.d/bash-profile-load.sh
