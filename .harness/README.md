# Harness configuration in this repo

**Read this before adding or changing anything under `.harness/`.**

## The one thing to know

> Adding a YAML file under `.harness/` does **not** create anything in Harness.

Harness entities in the `Custom_Models` / `datarobotusermodels` project fall into
two categories, and neither is "read from git on every push":

| Entity | Where it lives | What committing a file does |
| --- | --- | --- |
| Pipelines | Git (`storeType: REMOTE`) | Nothing until the pipeline is imported from git once. After that, Harness re-reads the file per run. |
| Input sets | Git (`storeType: REMOTE`) | Nothing until the input set is imported from git once. After that, Harness re-reads the file per run. |
| **Triggers** | **Inline in Harness only** | **Nothing, ever.** Harness never reads trigger YAML from git. The committed `*_on_pr.yaml` / `*_on_push_master.yaml` files are a statement of intent, not configuration. |

So a PR can add a complete and correct set of Harness YAML files, pass review,
merge to `master`, and change nothing at all about what actually runs. There is
no error, no warning, and no pipeline execution — the files simply sit there
looking authoritative. `enabled: true` in a committed trigger file means nothing
until someone creates that trigger in Harness by hand.

This is the trap that cost real time on
`public_dropin_environments/dr_mcp_execute_sandbox_minimal`: its trigger and
input-set YAMLs landed in #2137, were never registered, and its image has
therefore only ever been published manually.

Use `.harness/scripts/check_harness_entities.sh` to tell the difference; see
"Verifying" below.

## Publishing this image by hand

Until the triggers below are registered, the `dr_mcp_execute_sandbox_minimal`
image is published by running the `env_image_publish` pipeline yourself. It has
been published this way successfully several times; the pipeline itself is
sound.

Pipeline `env_image_publish`, org `Custom_Models`, project
`datarobotusermodels`. Inputs:

| Input | Value |
| --- | --- |
| codebase build | `branch` = `master` (or type `PR` with the PR number) |
| `context_path` | `public_dropin_environments/dr_mcp_execute_sandbox_minimal` |
| `dockerfile_path` | `public_dropin_environments/dr_mcp_execute_sandbox_minimal/Dockerfile` |
| `image_namespace_repo` | `datarobotdev/datarobot-user-models` |
| `image_tag` | `public_dropin_environments_dr_mcp_execute_sandbox_minimal_latest` (or `pr-<n>`) |

> **Never pass the `_latest` tag from a non-`master` branch.** `_latest` is a
> shared mutable tag that consumers pull directly. Building it from an unmerged
> branch overwrites the live image with unreviewed code. For anything that is
> not `master`, use `image_tag: pr-<n>` and a `PR`-type build.

These are exactly the values in the committed input sets
`dr_mcp_execute_sandbox_minimal_image_build_default_pr_input` (master/`_latest`)
and `dr_mcp_execute_sandbox_minimal_image_build_pr_input` (PR/`pr-<n>`) — once
those input sets are registered you can select one instead of retyping the
variables.

### Via the UI

*Pipelines → `env_image_publish` → Run*. Set the codebase build to branch
`master`, fill in the four pipeline variables above, and run. The
`BuildAndPushDockerRegistry` step reports the pushed tag in its logs.

### Via the API

```bash
curl -X POST \
  -H "x-api-key: ${HARNESS_PAT}" \
  -H 'Content-Type: application/yaml' \
  "https://app.harness.io/pipeline/api/pipeline/execute/env_image_publish\
?accountIdentifier=oP3BKzKwSDe_4hCFYw_UWA\
&orgIdentifier=Custom_Models\
&projectIdentifier=datarobotusermodels\
&moduleType=ci\
&branch=master" \
  --data-binary @- <<'YAML'
pipeline:
  identifier: env_image_publish
  properties:
    ci:
      codebase:
        build:
          type: branch
          spec:
            branch: master
  variables:
    - name: context_path
      type: String
      value: public_dropin_environments/dr_mcp_execute_sandbox_minimal
    - name: dockerfile_path
      type: String
      value: public_dropin_environments/dr_mcp_execute_sandbox_minimal/Dockerfile
    - name: image_namespace_repo
      type: String
      value: datarobotdev/datarobot-user-models
    - name: image_tag
      type: String
      value: public_dropin_environments_dr_mcp_execute_sandbox_minimal_latest
YAML
```

`branch=master` resolves the remote pipeline YAML; the `codebase.build` block
selects what gets checked out and built. For a PR build, swap the build block
for `type: PR` / `spec.number: <n>` and set `image_tag` to `pr-<n>`.

## Making it automatic: registering the committed triggers

The real fix for this environment is to register what is already committed.
Harness cannot pick these up from git, so this is a one-time manual step for
someone with write access to the project.

**Input sets** are git-backed, so import rather than retype. In the UI:
*Pipelines → `env_image_publish` → Input Sets → + New Input Set → Import From
Git*, pointing at the committed file path on `master`. Equivalent API call:

```bash
curl -X POST -H "x-api-key: ${HARNESS_PAT}" \
  "https://app.harness.io/pipeline/api/inputSets/import/<inputSetIdentifier>\
?accountIdentifier=oP3BKzKwSDe_4hCFYw_UWA\
&orgIdentifier=Custom_Models&projectIdentifier=datarobotusermodels\
&pipelineIdentifier=env_image_publish\
&connectorRef=account.svc_harness_git1&repoName=datarobot-user-models\
&branch=master&filePath=.harness/<file>.yaml"
```

**Triggers** are inline-only, so the committed YAML must be pasted in. In the
UI: *Pipelines → `env_image_publish` → Triggers → + New Trigger → GitHub →
Webhook*, then switch to the YAML editor and paste the committed file verbatim.

Afterwards, confirm the trigger shows as **Enabled** — a newly created trigger
is not enabled just because its YAML says `enabled: true` — then re-run
`check_harness_entities.sh` and confirm it reports `OK`.

Two things worth knowing before you do this:

* The committed conditions are sound. `changedFiles` and `targetBranch` both
  work on Push events in this project — `retag_push_docker_image` / `on_push`
  uses the same shape and fires reliably.
* All 21 existing `env_image_publish` triggers are disabled and none has fired
  since 2026-04-02, because environments that have an `env_info.json` moved to a
  generic mechanism keyed on that file. This environment deliberately has no
  `env_info.json` — it is a runtime sandbox image, not a catalog environment —
  so it needs its own triggers. Enabling one member of an otherwise-disabled
  family is intentional here, but worth a nod from the CI owners.

## Verifying that committed YAML is actually live

`.harness/scripts/check_harness_entities.sh` compares every committed trigger
and input-set YAML against the live Harness API. It is strictly read-only.

```bash
export HARNESS_PAT=<Harness personal or service-account token>
.harness/scripts/check_harness_entities.sh
```

It reports `OK` / `MISSING` / `DISABLED` per entity and exits non-zero if
anything committed here is missing from Harness.

### Known drift (August 2026)

It currently reports **19 missing** entities, so expect a non-zero exit until
that backlog is triaged:

* `dr_mcp_execute_sandbox_minimal` — both triggers and both input sets.
* `python312_local` — both triggers and all five input sets.
* Every `*_local_on_pr` trigger for the catalog environments (9 of them).

Some of that is dead config that should be deleted rather than registered.
Because of the backlog the checker is deliberately **not** wired into CI as a
blocking gate yet; doing that is the natural follow-up once it reads zero.

If you register or delete something, re-run the checker and update this section.

## File layout

Two layouts coexist. The newer, preferred one nests entities under their
pipeline:

```
.harness/pipelines/<pipelineIdentifier>/input_sets/<identifier>.yaml
```

The older flat layout (`.harness/<identifier>.yaml`) is still used by most
existing entities. Do not move an already-registered file: the registered entity
records its git `filePath`, and moving the file breaks it until it is
re-imported.
