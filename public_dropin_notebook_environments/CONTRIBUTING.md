# Contributing to the notebook environments

This directory holds the images behind DataRobot Codespaces and Notebooks:

* `python313_notebook` is the image Codespaces/Notebooks run today.
* `python311_notebook_base` is published as a template that customers extend to build their own notebook environments.

The `@datarobot/agent-lifecycle` team owns both images (see `.github/CODEOWNERS`).
Their review is required for every change under this directory.
Anyone is welcome to open a PR.

## Why review is required

Every package in these images becomes part of the vulnerability surface the owning team patches and ships.
Changes also reach customers: the 3.11 image is a public template, and new image versions are released to all notebook users.

## Before you open a PR

Say in the PR description which user need the change serves.

If your change adds, removes, or upgrades a **system package** (`apk add` in a `Dockerfile`):

- [ ] Applied to **both** images, or the PR explains why only one is affected.
- [ ] Built each affected image with `make image` from its directory.
- [ ] Ran the new tool inside each built image. Each image builds on a different `python-fips` tag (`3.11-dev` and `3.13-dev`), so a package that works on one is not proof it works on the other.
- [ ] Listed every package the change pulls in, including transitive ones, with the size increase.
- [ ] Ran a vulnerability scan on the built image and attached the result.
- [ ] Checked for setuid/setgid binaries and new listening ports or daemons.
- [ ] Decided on pinning: packages that have needed a CVE fix use a minimum version (for example `"task>=3.53.1-r4"`); say whether yours should too.

If your change adds a **long-running or background tool** (for example `tmux`, `sshd`, or a language server):

- [ ] Checked how it interacts with `agent/agent.py` and `agent/cgroup_watchers.py` (idle detection, resource accounting, cleanup on shutdown).
- [ ] Checked it works when the session runs as a different UID than the image was built with.

If your change touches the **3.11 template image**:

- [ ] Considered customers who extend it. Anything added becomes part of their base.

If your change adds a **Python dependency**:

- [ ] Edited `requirements.txt` for the image, keeping existing pins consistent.

## What not to edit by hand

* `environmentVersionId` and `tags` in `env_info.json`. CI pushes a version-bump commit to your PR branch.
* Requirements files in other `public_dropin_environments/*` directories. CI regenerates them, so they can appear in your diff.

## Review

`@datarobot/agent-lifecycle` decides whether the change is merged.
If your change affects the notebook user experience, say so in the PR description so reviewers can weigh it.
