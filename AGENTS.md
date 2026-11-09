# Repository workflow

- CLI and GUI are independent repositories; do not merge their source trees.
- Keep `main` as the only local and remote development branch. Work directly on `main`; do not create feature branches or worktrees unless the user explicitly requests one.
- Use one agent and work sequentially. Preserve Git history; do not force-push.
- Keep personal settings, model weights, training outputs and datasets out of commits.
