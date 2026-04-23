import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from eecs148b_hw1.models.transformer import Transformer
from eecs148b_hw1.data.loader import get_batch
from eecs148b_hw1.utils.loss import cross_entropy, perplexity

def load_data(path: str, dtype=np.uint16):
    if path.endswith(".npy"):
        data = np.load(path, mmap_mode="r")
    else:
        data = np.memmap(path, dtype=dtype, mode="r")

    return data


def save_model(model, loss, optimizer, step, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"model_{step}.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
        'args': vars(args),
    }, path)
    print(f"Checkpoint saved to {path}")


def evaluate(model, data, batch_size, context_length, device, eval_iters=100):
    model.eval()
    losses = []

    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size, context_length, device)
            logits = model(x)
            loss = cross_entropy(logits, y)
            losses.append(loss.item())
    
    model.train()
    avg_loss = torch.tensor(losses).mean()
    ppl = torch.exp(avg_loss)

    return avg_loss, ppl


def train(model, optimizer, args, device, *, wandb_run=None):
    train_data = load_data(args.train_data)
    val_data = load_data(args.val_data)

    model.train()
    x_fixed = None
    y_fixed = None
    if args.overfit_batch:
        x_fixed, y_fixed = get_batch(train_data, args.batch_size, args.context_length, device)

    pbar = tqdm(range(1, args.max_steps + 1), desc="Training", unit="step")
    for step in pbar:
        if x_fixed is None:
            x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        else:
            x, y = x_fixed, y_fixed
        logits = model(x)
        loss = cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if step % args.log_every == 0:
            tqdm.write(f"Step {step}/{args.max_steps} | train_loss: {loss.item():.4f}")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": loss.item(),
                        "train/ppl": float(torch.exp(loss.detach()).item()),
                        "lr": optimizer.param_groups[0].get("lr", None),
                    },
                    step=step,
                )

        if step % args.eval_every == 0:
            val_loss, val_ppl = evaluate(
                model,
                val_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=device,
            )
            tqdm.write(f"Step {step}/{args.max_steps} | val_loss: {val_loss.item():.4f} | val_ppl: {val_ppl:.4f}")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "val/loss": float(val_loss.item()),
                        "val/ppl": float(val_ppl),
                    },
                    step=step,
                )

        if step % args.save_every == 0:
            save_model(model, loss, optimizer, step, args.ckpt_dir)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train a transformer model")
    
    # Data
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)

    # Model
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)

    # Training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # logging and checkpointing
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=10)

    # wandb
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="eecs148b-hw1", help="W&B project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Optional W&B run name.")

    # debugging
    parser.add_argument(
        "--overfit_batch",
        action="store_true"
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_model * 4,
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except Exception as e:
            raise SystemExit(
                "W&B logging requested (--wandb) but `wandb` is not available. "
                "Install it or disable --wandb."
            ) from e
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"{args.wandb_run_name}-B={args.batch_size},C={args.context_length},D={args.d_model},H={args.num_heads},L={args.num_layers},V={args.vocab_size},T={args.max_steps}",
            config=vars(args),
        )

    try:
        train(model, optimizer, args, device, wandb_run=wandb_run)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
