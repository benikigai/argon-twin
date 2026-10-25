"""Prepared Gemma 4 E4B QLoRA pilot. Run on a separate Colab GPU after reconstruction.

Install unsloth, datasets and trl using Unsloth's Gemma 4 Colab instructions.
Upload runs/gemma-pilot/{train,eval}.jsonl to /content/gemma-pilot first.
This saves an adapter, not a published or automatically deployed model.
"""
import json
from pathlib import Path

import torch
from unsloth import FastModel
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

DATA = Path('/content/gemma-pilot')
OUTPUT = Path('/content/argon-twin-gemma-e4b-adapter')
MODEL = 'unsloth/gemma-4-E4B-it-unsloth-bnb-4bit'
MODEL_REVISION = '6f10b2aa522ba471321d8246fd71a3dfa8c5b9a2'

rows = {split: [json.loads(line) for line in (DATA / f'{split}.jsonl').read_text().splitlines()]
        for split in ['train', 'eval']}
assert rows['train'] and rows['eval'], 'Need both measured training and held-out evaluation pairs'
assert not ({r['scene_group'] for r in rows['train']} & {r['scene_group'] for r in rows['eval']})
assert all(r['evidence']['repaired_passed'] for r in rows['train'])

model, tokenizer = FastModel.from_pretrained(model_name=MODEL, revision=MODEL_REVISION, max_seq_length=2048,
    load_in_4bit=True, full_finetuning=False)
model = FastModel.get_peft_model(model, finetune_vision_layers=False,
    finetune_language_layers=True, finetune_attention_modules=True,
    finetune_mlp_modules=True, r=16, lora_alpha=16, lora_dropout=0,
    bias='none', random_state=3407, use_gradient_checkpointing='unsloth')
datasets = {split: Dataset.from_list([{'text': tokenizer.apply_chat_template(
    row['messages'], tokenize=False, add_generation_prompt=False, enable_thinking=False)}
    for row in values]) for split, values in rows.items()}
trainer = SFTTrainer(model=model, processing_class=tokenizer,
    train_dataset=datasets['train'], eval_dataset=datasets['eval'],
    args=SFTConfig(output_dir=str(OUTPUT), dataset_text_field='text', max_length=2048,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=2e-4, max_steps=60, warmup_steps=5, logging_steps=5,
        optim='adamw_8bit', weight_decay=.01, lr_scheduler_type='linear',
        seed=3407, report_to='none', fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(), save_strategy='no'))
trainer = train_on_responses_only(trainer, instruction_part='<|turn>user\n',
                                 response_part='<|turn>model\n')
result = trainer.train()
model.save_pretrained(str(OUTPUT))
tokenizer.save_pretrained(str(OUTPUT))
evaluation = trainer.evaluate()
(OUTPUT / 'training-receipt.json').write_text(json.dumps({
    'model': MODEL, 'revision': MODEL_REVISION, 'train_pairs': len(rows['train']), 'eval_pairs': len(rows['eval']),
    'train_metrics': result.metrics, 'eval_metrics': evaluation,
    'physics_benchmark_pending': True,
    'warning': 'Loss is not a hand-over success metric; compare base and adapter proposals in MuJoCo.'
}, indent=2))

# Compare adapters on the identical held-out prompts; physics judging runs locally.
from contextlib import nullcontext
FastModel.for_inference(model)
predictions = []
for label in ["base", "tuned"]:
    context = model.disable_adapter() if label == "base" else nullcontext()
    with context:
        for index, row in enumerate(rows["eval"]):
            batch = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=True,
                return_tensors="pt").to("cuda")
            import time
            started = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(**batch, max_new_tokens=192, do_sample=False)
            answer = tokenizer.decode(generated[0][batch["input_ids"].shape[-1]:], skip_special_tokens=True)
            predictions.append({"model": label, "scene_group": row["scene_group"],
                "input": row["messages"][-2]["content"], "prediction": answer,
                "latency_s": time.monotonic() - started})
            (OUTPUT / "predictions.json").write_text(json.dumps(predictions, indent=2))
            print("PREDICTION", label, index + 1, len(rows["eval"]), flush=True)
(OUTPUT / "predictions.json").write_text(json.dumps(predictions, indent=2))

import shutil
shutil.make_archive('/content/argon-twin-gemma-e4b', 'gztar', OUTPUT)
print('GEMMA_PILOT_COMPLETE', flush=True)
