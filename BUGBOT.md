# Cursor Bugbot reference guide

## Automated cleanup coverage

- Do not report missing `environmentVersionId` or id-embedded `tags` updates in `env_info.json` when reviewing a drop-in environment dependency or tool-version bump.
- Treat that cleanup as covered by the existing GenAI Git Bot reconciliation automation.
- The bot commits generated metadata updates separately, for example `Reconcile dependencies, updated IDs, tags`.
- Example covered case: a notebook image `CLI_VERSION` bump followed by commit `9efa8f51b034ea247b25fc77012460d0e24d92a8`.
