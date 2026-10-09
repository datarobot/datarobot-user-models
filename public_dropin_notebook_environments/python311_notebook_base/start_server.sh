#!/bin/bash

export HOME="/home/notebooks"

# setup the working directory for the kernel
if [ -z "$1" ]; then
    # Set default working directory if no argument is provided
    WORKING_DIR="/home/notebooks"
else
    # Use the provided working directory
    WORKING_DIR="$1"
fi

export WORKING_DIR

# FLEET-8918: a hardened cluster can run this session at a uid the image was not built with,
# which leaves $HOME and every image layer read-only, so each redirect below is gated on $HOME
# actually being unwritable and a default install keeps the paths it always had.
export NOTEBOOKS_RW_DIR="${NOTEBOOKS_RW_DIR:-/home/notebooks/.nbx-rw}"

if [ -w "$HOME" ]; then
    NBX_HOME_WRITABLE=true
else
    NBX_HOME_WRITABLE=false
    mkdir -p "${NOTEBOOKS_RW_DIR}/profile.d" || \
        echo "WARNING: neither ${HOME} nor ${NOTEBOOKS_RW_DIR} is writable - session set-up will be degraded." >&2

    # git >= 2.32, IPython >= 7 and Jupyter all read these, so none of them needs a writable $HOME.
    export GIT_CONFIG_GLOBAL="${NOTEBOOKS_RW_DIR}/gitconfig"
    export IPYTHONDIR="${NOTEBOOKS_RW_DIR}/ipython"
    export JUPYTER_CONFIG_DIR="${NOTEBOOKS_RW_DIR}/jupyter"
    export JUPYTER_DATA_DIR="${NOTEBOOKS_RW_DIR}/jupyter/data"
    export JUPYTER_RUNTIME_DIR="${NOTEBOOKS_RW_DIR}/jupyter/runtime"
    # matplotlib, pip and git's own credential cache all write under $HOME through these, so
    # they move too or every kernel importing matplotlib rebuilds its font cache in /tmp.
    export XDG_CONFIG_HOME="${NOTEBOOKS_RW_DIR}/config"
    export XDG_CACHE_HOME="${NOTEBOOKS_RW_DIR}/cache"
    mkdir -p "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME" 2>/dev/null || true
    touch "$GIT_CONFIG_GLOBAL" 2>/dev/null || true

    # The image bakes PYTHONPATH=/home/notebooks/.ipython/extensions, so a moved IPYTHONDIR has
    # to be added or dataframe_formatter stops importing and dataframe rendering disappears.
    export PYTHONPATH="${IPYTHONDIR}/extensions:${PYTHONPATH}"
fi
export NBX_HOME_WRITABLE

# FLEET-8918: sshd does not run as root here, so it can only finish a login when the user the
# client authenticates as already resolves to the uid the session runs at, which an assigned
# uid never does; point the image's own user at it so terminals and VS Code keep working.
NBX_IMAGE_UID="$(awk -F: '$1 == "notebooks" { print $3; exit }' /etc/passwd)"
if [ "$(id -u)" != "$NBX_IMAGE_UID" ]; then
    NBX_PASSWD_NEW="${NOTEBOOKS_RW_DIR}/passwd"
    if [ -w /etc/passwd ] \
        && awk -F: -v uid="$(id -u)" 'BEGIN { OFS=":" } $1 == "notebooks" { $3 = uid } { print }' \
            /etc/passwd > "$NBX_PASSWD_NEW" 2>/dev/null \
        && [ "$(wc -l < "$NBX_PASSWD_NEW")" -eq "$(wc -l < /etc/passwd)" ]; then
        # Truncate in place rather than replace: the file is writable, /etc is not.
        cat "$NBX_PASSWD_NEW" > /etc/passwd
    else
        echo "WARNING: uid $(id -u) has no /etc/passwd entry and /etc/passwd cannot be updated, so ssh logins (terminals and VS Code) will fail." >&2
    fi
    rm -f "$NBX_PASSWD_NEW"
fi

VERBOSE_MODE=true
# shellcheck disable=SC1091
source /etc/system/kernel/setup-venv.sh $VERBOSE_MODE

cd /etc/system/kernel/agent || exit
nohup uvicorn agent:app --host 0.0.0.0 --port 8889 &

# shellcheck disable=SC1091
source /etc/system/kernel/common-user-limits.sh

# shellcheck disable=SC1091
source /etc/system/kernel/setup-ssh.sh
cp -L /var/run/notebooks/ssh/authorized_keys/notebooks /etc/authorized_keys/ && chmod 600 /etc/authorized_keys/notebooks
# mkdir -p, not mkdir: an emptyDir mounted over this path makes a plain mkdir fail with
# "File exists", so the cp never runs and sshd comes up with no host key while the pod
# still reports 1/1 Running.
mkdir -p /etc/ssh/keys && cp -L /var/run/notebooks/ssh/keys/ssh_host_* /etc/ssh/keys/ && chmod 600 /etc/ssh/keys/ssh_host_*
# StrictModes refuses an authorized_keys file whose directory group or others can write, which
# is what an emptyDir mounted over /etc/authorized_keys is, so the check is dropped only when
# that is really the case and a default install keeps it.
NBX_SSHD_OPTS=()
if find /etc/authorized_keys -maxdepth 0 \( -perm -0020 -o -perm -0002 \) 2>/dev/null | grep -q .; then
    echo "WARNING: /etc/authorized_keys is writable by group or others, so sshd runs with StrictModes off." >&2
    NBX_SSHD_OPTS=(-o StrictModes=no)
fi
nohup /usr/sbin/sshd -D ${NBX_SSHD_OPTS[@]+"${NBX_SSHD_OPTS[@]}"} &

# Ensure proper permissions on the directory used by cache daemon to ensure it starts (create dir. if needed)
if mkdir -p /home/notebooks/storage/.cache/git/credential 2>/dev/null; then
    chmod 700 /home/notebooks/storage/.cache/git/credential
else
    echo "WARNING: /home/notebooks/storage is not writable, so the git credential cache daemon may not start." >&2
fi
# Initialize the git helper. Features are turned on/off dependent on `GITHELPER_*` env vars
# drgithelper writes two dotfiles into $HOME with no flag to move them, so only an unwritable
# $HOME gets a redirected one and a default install keeps its git credential cache across
# a session restart.
if [ "$NBX_HOME_WRITABLE" = true ]; then
    /etc/system/kernel/drgithelper configs set
else
    NBX_TOOL_HOME="${NOTEBOOKS_RW_DIR}/home"
    mkdir -p "$NBX_TOOL_HOME" 2>/dev/null || true
    HOME="$NBX_TOOL_HOME" /etc/system/kernel/drgithelper configs set
    # Kernels and terminals keep HOME=/home/notebooks, so carry the redirect into the helper
    # line itself or every git credential call writes its log to the read-only $HOME.
    sed -i -E "s|^([[:space:]]*helper[[:space:]]*=[[:space:]]*)!?(.*drgithelper.*)$|\1!HOME=${NBX_TOOL_HOME} \2|" \
        "$GIT_CONFIG_GLOBAL"
fi

# no trailing slash in the working dir path
git config --global --add safe.directory "${WORKING_DIR%/}"

# setup the working directory for the kernel
cd "$WORKING_DIR" || exit

# setup ipython extensions
# `/etc/ipython/.` rather than `/etc/ipython/`: the trailing-slash form nests a second
# ipython/ directory under the target on any re-run.
NBX_IPYTHONDIR="${IPYTHONDIR:-${HOME}/.ipython}"
mkdir -p "${NBX_IPYTHONDIR}" && cp -r /etc/ipython/. "${NBX_IPYTHONDIR}/"

# clear out kubernetes_specific env vars before starting kernel gateway as it will inherit them
prefix="KUBERNETES_"; for var in $(printenv | cut -d= -f1); do [[ "$var" == "$prefix"* ]] && unset "$var"; done

exec jupyter kernelgateway --config=/etc/system/kernel/jupyter_kernel_gateway_config.py --debug
