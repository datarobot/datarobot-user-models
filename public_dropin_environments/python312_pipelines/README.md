# [DataRobot] Python 3.12 Pipelines

Base image for DataRobot Pipelines images on Python 3.12. pipelines-api builds each user-defined
pipeline image on top of this environment: it copies its runner scripts to `/opt/covalent`, installs the
user's packages into the venv at `/var/lib/covalent` against `constraints.txt`, and runs the prep, task
and schedule-trigger pods on the result. This environment is not meant to be selected for custom models,
jobs, notebooks or applications.

## What is in the image

- Python 3.12 on the DataRobot Chainguard FIPS base, venv at `/var/lib/covalent` (first on `PATH`).
- The pipelines runtime set from [requirements.in](requirements.in): covalent-cloud, the DataRobot SDK
  with the `pipelines` extra, boto3, azure-storage-blob, azure-identity, google-cloud-storage, uv,
  psutil, PyYAML. Exact pins: [requirements.txt](requirements.txt).
- A writable git clone of cloudpickle at `/var/lib/covalent/bin/cloudpickle`; the runner checks out the
  version the client pickled with at task start.
- `git`, busybox and `sh`; `constraints.txt` (a `pip freeze` without the datarobot and covalent lines).
- `/opt/pipelines-env/smoke_test.py`, the contract check: run it inside the image and expect exit 0.

No entrypoint: pipelines-api sets the command on every pod.
