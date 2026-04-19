import argparse


args = parser.parse_args()

def load_data(path: str, dtype=np.uint16):
    if path.endswith(".npy"):
        data = np.load(path, mmap_mode="r")
    else:
        data = np.memmap(path, dtype=dtype, mode="r")

    return data


def save_model(model, save_dir, step):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    path = os.path.join(args.ckpt_dir, f"model_{step}.pt")
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
    perplexity = perplexity(torch.tensor(losses), context_length)

    return avg_loss, perplexity


def train(model, optimizer, args):
    train_data = load_data(args.train_data)
    val_data = load_data(args.val_data)

    model.train()
    for step in range(1, args.max_steps + 1):
        x, y = get_batch(train_data, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0:
             print(f"Step {step}/{args.max_steps} | train_loss: {loss.item():.4f} | train_ppl: {ppl:.4f}")

        if step % args.eval_every == 0:
            val_loss, val_ppl = evaluate(model, val_data, batch_size, context_length, device)
            print(f"Step {step}/{args.max_steps} | val_loss: {val_loss.item():.4f} | val_ppl: {val_ppl:.4f}")

        if step % args.save_every == 0:
            save_model(model, args.save_dir, step)
            
parser = argparse.ArgumentParser()
parser.add_argument("--num-epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--context-length", type=int, default=128)
parser.add_argument("--d-model", type=int, default=128)
parser.add_argument("--num-layers", type=int, default=6)
parser.add_argument("--num-heads", type=int, default=8)
parser.add_argument("--d-ff", type=int, default=128)
parser.add_argument("--save-dir", type=str, default="checkpoints")
parser.add_argument("--save-every", type=int, default=10)

print(parser.parse_args())

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

    args = parser.parse_args()
    
    self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int

    model = Transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_model * 4,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    train(model, optimizer, args)
