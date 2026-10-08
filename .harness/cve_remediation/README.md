# CVE Remediation pipeline

The same pipeline `global-envs-models` runs, pointed at this repo. Its shape — one stage,
triage → report → agent → PR → artifacts, and why it clones by hand rather than using the
built-in checkout — is documented once, in
[`global-envs-models/.harness/cve_remediation/README.md`](https://github.com/datarobot/global-envs-models/blob/main/.harness/cve_remediation/README.md).
This file covers only what is different here.

## This repo already automates the mechanical half

`pdie_update_deps_versions` ("PDIE update versions and dependencies") regenerates
`requirements.txt` and bumps `environmentVersionId` for every changed env under
`public_dropin_environments/`, and auto-commits onto the PR branch. The app-charts bump
follows it. None of that needs an agent.

So the agent is not here to do that work again. It is here for what the automation leaves:
a ticket still open once the automation has had its run, which needs either a justification
or a fix the regen cannot make.

### The grace window

`grace_days` (default 3) splits the inventory. A ticket younger than that is **deferred** —
logged by key and age in the Triage step, not acted on — because the automation is very likely
already handling it, and a second actor racing it costs a conflicting PR. Older tickets go
to the agent.

Age comes from the raw Jira payload, which carries `created`; the inventory row does not.
**An unparseable or missing timestamp counts as aged**, so a ticket is never hidden by a bad
date. A run where everything is deferred stops with that reason stated — it is a quiet day,
not a failure, and it says so differently from "no ticket matched".

### Open PRs, because the window is not the whole story

The grace window says the automation **had its run**. It does not say the result **landed**. The
regen PRs — `[auto] Regen dependencies for all dropin envs` from `svc-harness-git2`, and
`[RAPTOR-...] Regen requirements` from `nullspoon` — are not reliably merged: two were closed
unmerged on 2026-10-01 and one sat open from 2026-10-05. A ticket whose fix is parked in one of
those is still Open in Jira, so it ages past the window and reaches the agent looking untouched.

So Triage also stages `open-prs.json`: the open PRs on this ref that touch
`public_dropin_environments/`, with the paths each changes. The playbook reads it before editing an
env. The paths are the point — the auto-regen PRs carry no ticket key, so the playbook's existing
search by key cannot see them.

### What this pipeline does not decide

Whether PDIE covers the bump and regen on a given ref is a standing fact about the repo, not
something this pipeline measures: it runs on `master` and not on `release/11.1`. That rule lives
in the agent's playbook (`packs/raptor/playbooks/datarobot-user-models.md`) and is maintained by
hand when the workflow changes. The pipeline stages only what the agent cannot cheaply get for
itself — the VITA scan, the Jira inventory, the built tags — and the agent has a full checkout to
read the rest from.

**PDIE does not skip the envs without a `requirements.in`.** `pdie_update_deps_versions` iterates
every changed `public_*` environment with no such gate: it bumps `environmentVersionId` and runs
`make update-deps`, which for `python3_mcp` and `python3_genai_agents` is `uv lock --upgrade`.
They are named in the Triage log only because the regen mechanism differs, so a diff on them looks
unlike the others — not because they are left alone. Treating them as untouched is how the agent
would pin one and then have PDIE auto-commit on top of its branch, which is the race the grace
window exists to prevent.

## What this pipeline cannot close

Requirements regen is pip-level. A Chainguard base OS package — glibc, busybox, libexpat —
has no row in `requirements.txt`, so neither PDIE nor the agent moves it by regenerating.
Those need a rebuild once the base ships a fix, and stay on the justification path until
then. A run that leaves them open has not failed.

## Running it

`mode` defaults to `shadow`, which denies every remote mutation. Promote deliberately:
`shadow` → `dry-run` → `hitl`, reading the RunReport at each rung.
