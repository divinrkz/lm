import argparse

import torch

from eecs148b_hw1.tokenizer.tokenizer import BPETokenizer
from eecs148b_hw1.models.transformer import Transformer
from eecs148b_hw1.utils.functional import Functional as F


def generate(
    model: torch.nn.Module,
    tokenizer: BPETokenizer,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    context_length: int,
    device: str = "cpu",
) -> str:
    model.eval()
    input_ids = tokenizer.encode(prompt)
    tokens = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    eot_ids = tokenizer.encode("<|endoftext|>")
    eos_id = eot_ids[0] if eot_ids else None

    with torch.no_grad():
        for _ in range(max_tokens):
            x = tokens[:, -context_length:] if tokens.size(1) > context_length else tokens
            logits = model(x)
            logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else: 
                logits = logits / temperature

                if top_p < 1.0:
                    logits = top_p_filter(logits, top_p)
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            tokens = torch.cat([tokens, next_token], dim=1)
            if eos_id is not None and next_token.item() == eos_id:
                break

    output_ids = tokens[0].tolist()

    return tokenizer.decode(output_ids)

def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Zero out tokens outside the nucleus (smallest set whose cumulative prob >= p)."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Find where cumulative prob exceeds p, shift right so the boundary token is kept
    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
    sorted_logits[sorted_mask] = float("-inf")

    # Scatter back to original ordering
    logits = torch.zeros_like(logits)
    logits.scatter_(1, sorted_indices, sorted_logits)
    return logits


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, default="bpe/tinystories_vocab.json")
    parser.add_argument("--merges_path", type=str, default="bpe/tinystories_merges.txt")
    parser.add_argument("--context_length", type=int, default=128)

    # wandb
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="eecs148b-hw1", help="W&B project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Optional W&B run name.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BPETokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=["<|endoftext|>"])
    vocab_size = len(tokenizer.vocab)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    print(ckpt_args)
    model = Transformer(
        vocab_size=int(ckpt_args.get("vocab_size", vocab_size)),
        context_length=int(ckpt_args.get("context_length", args.context_length)),
        d_model=int(ckpt_args.get("d_model", 512)),
        num_layers=int(ckpt_args.get("num_layers", 4)),
        num_heads=int(ckpt_args.get("num_heads", 8)),
        d_ff=int(ckpt_args.get("d_ff", 512*4)),
    ).to(device)
    model.load_state_dict(state, strict=False)

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
            name=args.wandb_run_name,
            config={
                **vars(args),
                "checkpoint_step": ckpt.get("step", None) if isinstance(ckpt, dict) else None,
                "checkpoint_args": ckpt_args,
            },
        )

    text = generate(
        model,
        tokenizer,
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        context_length=args.context_length,
        device=str(device),
    )

    print("-" * 100)
    print(f"prompt: {args.prompt}")
    print(f"response: {text}")
    print("-" * 100)
    if wandb_run is not None:
        wandb_run.log(
            {
                "prompt": args.prompt,
                "generated_text": text,
                "generation": wandb.Table(data=[[args.prompt, text]], columns=["prompt", "generated"]),
            }
        )
        wandb_run.finish()

