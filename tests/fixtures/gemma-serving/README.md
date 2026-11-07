# Gemma serving test fixtures

These frozen inputs let the existing mocked Gemma client tests run without a workstation Models directory. The workflow sets `ARGON_GEMMA_SERVING_DIR` to this folder. Production defaults are unchanged.

`repair_client.py` contains only the exact `SYSTEM` constant extracted from the serving client. `plan-patch.schema.json` is copied unchanged. Neither file contains model weights, credentials, or evaluation results. These tests validate client request and error handling, not a live model or adapter.

Source snapshots taken November 7, 2026:

- repair_client.py source SHA-256: `9e1792032cd387a709b168cec88f78942e46ff9c8b555ee666275b8846bfd215`
- plan-patch.schema.json source SHA-256: `f236c19664faabbb8c9bfbe99c2b767b891b7e334d73536bd9c2421fa35077f9`
