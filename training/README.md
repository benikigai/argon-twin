# Gemma repair-model pilot

Selected student: `google/gemma-4-E4B-it`; training loader: `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit`
(pinned revision `6f10b2aa522ba471321d8246fd71a3dfa8c5b9a2`).
Use 4-bit QLoRA, rank 16, 2,048-token sequences, batch size 1 and gradient accumulation 4.
Train the language adapter only. The first pilot consumes structured scene, plan and measured
failure information and returns a `PlanPatch`. It does not control motors or replace MuJoCo's judge.

The 2026-10-25 T4 pilot completed 60 training steps. On the same 12 held-out cases
from four scene variants, the Colab base model passed 11/12 MuJoCo checks and the tuned
adapter passed 12/12. Both produced 12/12 valid patches. Median inference was 4.29 s
for base and 5.46 s for tuned. This small pilot does not establish a general advantage.
Receipts: `runs/gemma-evaluation/colab-paired/`. Local adapter serving remains unverified.

## Run

1. From the repository root, run `.venv/bin/python training/collect_repair_pairs.py`.
   Outputs go under the gitignored `runs/gemma-pilot/`. Labels are scripted corrections
   whose repaired plan passed the actual simulator; they are not attributed to Argon.
2. Review the dataset receipt and keep scene groups 12–15 held out. Run the base model
   on these same held-out prompts as the tuned adapter. Do not train on evaluation labels.
3. After the splat job, install the current Gemma 4 dependencies from the official
   [Unsloth training guide](https://unsloth.ai/docs/models/gemma-4/train) in a separate Colab runtime.
   Upload `train.jsonl`, `eval.jsonl`, and `training/finetune_colab.py`.
4. Run the prepared tuning script. It writes a local adapter and a training receipt.
5. Download its base/adapter prediction file and evaluate it with
   `.venv/bin/python training/evaluate_patches.py predictions.json`.
   Report JSON validity, held-out MuJoCo task success, and inference latency.

The small dataset is a synthetic-scene pilot, not evidence of broad robotic competence.
Training loss alone does not demonstrate better hand-overs. Keep base Gemma if the adapter
does not improve held-out task success. Any local/offline serving claim also needs an actual
export and inference test on the target machine.

## Argon's long-decode role

After the core integration works, give Argon the scene/controller contracts and measured
training-case evidence in one request. Ask for one consistent package containing a scene
description, controller/test changes, case-specific diagnoses and candidate repair labels.
Use `background=True`, high thinking and an output ceiling of 262,144 tokens. Keep held-out
scene groups out of the teacher request. Execute generated code only in the managed sandbox.

Every proposed repair must run through the real simulator before it becomes a training label.
Raw positions, contacts and trajectories come from MuJoCo, never from invented text. Record
the interaction ID and API-reported input, output and thinking counts separately. Claim an
output beyond 65,536 only when the measured output itself crosses that threshold. Do not pad
the package to reach a quota.

Sources: [Google model card](https://huggingface.co/google/gemma-4-E4B-it),
[Google memory guide](https://ai.google.dev/gemma/docs/core), and
[Unsloth's Gemma 4 training guide](https://unsloth.ai/docs/models/gemma-4/train).
