# Design Discussions

This file is intentionally kept short.

The repo’s **canonical training stack** is fully in-repo (no external trainer integration):
- Entry point: `python -m training.run`
- Algorithms: `ppo_smdp` and `simple_pg`

For current design decisions, risks, and roadmap, use:
- `docs/DESIGN.md`
- `notes/MEMORY.md` (decisions/ADR-lite)
- `notes/SCRUTINY.md` (risks + validation plans)
- `notes/BACKLOG.md` (prioritized work)
- `notes/WORKLOG.md` (chronological work log)
