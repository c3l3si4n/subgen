"""
Training entrypoint for the subdomain generation model.

Uses HuggingFace Trainer with cosine LR schedule, BF16, wandb logging.
"""

import argparse
import os

from transformers import (
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from data.dataset import DomainDataset, PackedDataCollator
from model.config import build_config, build_model


def main():
    parser = argparse.ArgumentParser(description="Train subdomain generation model")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from checkpoint path")

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--per-device-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=100)

    # Precision and hardware
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-torch-compile", action="store_true",
                        help="Disable torch.compile")

    # Logging
    parser.add_argument("--wandb-project", type=str, default="subgen")
    parser.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args()

    use_bf16 = args.bf16 and not args.no_bf16

    # Work around PyTorch Inductor TF32 API conflict (pytorch/pytorch#166387)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)

    # Build model — use FA2 if flash-attn is installed, else SDPA fallback
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        print("Using FlashAttention 2 (optimized CUDA kernels)")
    except ImportError:
        attn_impl = "sdpa"
        print("flash-attn not installed, falling back to SDPA")
    config = build_config(vocab_size=len(tokenizer), attn_implementation=attn_impl)
    model = build_model(config)

    # Load datasets
    train_dataset = DomainDataset(os.path.join(args.data_dir, "train.bin"))
    val_dataset = DomainDataset(os.path.join(args.data_dir, "val.bin"))
    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences: {len(val_dataset)}")

    # Training arguments
    report_to = "none" if args.no_wandb else "wandb"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        adam_beta1=0.9,
        adam_beta2=0.95,
        lr_scheduler_type="cosine",
        bf16=use_bf16,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=report_to,
        run_name="subgen-150m",
        optim="adamw_torch_fused",
        dataloader_num_workers=8,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        gradient_checkpointing=attn_impl != "flash_attention_2",
        torch_compile=not args.no_torch_compile,
        label_smoothing_factor=0.05,
    )

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=PackedDataCollator(use_fa2=attn_impl == "flash_attention_2"),
        processing_class=tokenizer,
    )

    # Train
    resume_from = args.resume_from
    if resume_from is None and os.path.isdir(args.output_dir):
        # Auto-detect latest checkpoint
        checkpoints = [
            d for d in os.listdir(args.output_dir)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            resume_from = os.path.join(args.output_dir, latest)
            print(f"Auto-resuming from {resume_from}")

    trainer.train(resume_from_checkpoint=resume_from)

    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
