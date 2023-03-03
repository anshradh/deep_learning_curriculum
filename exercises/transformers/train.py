from argparse import ArgumentParser, Namespace
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import GPT, GPTConfig
from mpi4py import MPI
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import einops


def train(args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    torch.manual_seed(args.seed + rank)

    device = torch.device(args.device, rank)

    model_config = GPTConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_ctx=args.n_ctx,
        d_vocab=args.d_vocab,
        d_vocab_out=args.d_vocab_out,
        use_mlp=args.use_mlp,
        use_norm=args.use_norm,
        d_mlp=args.d_mlp,
        norm_eps=args.eps,
        act_fn=F.gelu if args.act_fn == "gelu" else F.relu,
        weight_std=args.weight_std,
        device=device,
    )

    model = GPT(model_config).to(device)

    batch_size = args.batch_size // args.device_count

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if rank == 0 and args.use_wandb:
        import wandb

        wandb.init(project="lm_training", config=args.__dict__)

    hooks = []
    for p in model.parameters():
        comm.Bcast(p.data, root=0)
        hooks.append(
            p.register_hook(lambda grad: grad / size if grad is not None else grad)
        )

    comm.Barrier()

    if args.data is not None:
        dataset = torch.load(args.data)

        local_dataset = dataset[
            rank * len(dataset) // size : (rank + 1) * len(dataset) // size
        ]

        for epoch in tqdm(range(1, args.epochs + 1), desc="Epoch {epoch}"):
            for i in tqdm(
                range(0, len(local_dataset) // batch_size),
            ):
                optimizer.zero_grad()

                batch = local_dataset[i * batch_size : (i + 1) * batch_size].to(device)

                inputs = batch[:, :-1]
                targets = batch[:, 1:]

                logits = model(inputs)

                logits = einops.rearrange(
                    logits,
                    "batch seq vocab -> (batch seq) vocab",
                )
                preds = torch.argmax(logits, dim=-1)
                targets = einops.rearrange(targets, "batch seq -> (batch seq)")

                loss = F.cross_entropy(
                    logits,
                    targets,
                    ignore_index=0,
                )
                acc = (preds == targets).float().mean()

                loss.backward()

                for p in model.parameters():
                    if p.grad is not None:
                        comm.Allreduce(MPI.IN_PLACE, p.grad, op=MPI.SUM)

                comm.Barrier()

                optimizer.step()

                if args.use_wandb and rank == 0:
                    wandb.log(
                        {
                            "loss": loss.item(),
                            "accuracy": acc.item(),
                            "epoch": epoch,
                            "batch": i,
                            "examples": (i + 1) * batch_size
                            + (epoch - 1) * len(dataset),
                        }
                    )

            if (epoch - 1) % args.print_interval == 0 and rank == 0:
                print(
                    f"Rank {rank} - Epoch {epoch} - Loss {loss.item()} - Accuracy {acc.item()}"
                )

            if (epoch - 1) % args.save_interval == 0 and rank == 0:
                torch.save(model.state_dict(), args.save_path)

    else:
        for epoch in tqdm(range(1, args.epochs + 1), desc="Epoch {epoch}"):
            if args.task == "reverse_seq":
                first_half = torch.randint(
                    0,
                    args.d_vocab,
                    (batch_size, args.n_ctx // 2),
                    device=device,
                )
                second_half = torch.flip(first_half, dims=[1])
                batch = torch.cat([first_half, second_half], dim=1)
                inputs = batch[:, :-1].clone()
                targets = batch[:, 1:].clone()

                targets[:, : args.n_ctx // 2 - 1] = -1

                logits = model(inputs)

                preds = torch.argmax(logits, dim=-1)
                acc = (
                    (
                        preds[:, args.n_ctx // 2 - 1 :]
                        == targets[:, args.n_ctx // 2 - 1 :]
                    )
                    .float()
                    .mean()
                )
            else:
                raise NotImplementedError

            logits = einops.rearrange(
                logits,
                "batch seq vocab -> (batch seq) vocab",
            )
            targets = einops.rearrange(targets, "batch seq -> (batch seq)")
            loss = F.cross_entropy(
                logits,
                targets,
                ignore_index=-1,
            )

            loss.backward()

            for p in model.parameters():
                if p.grad is not None:
                    comm.Allreduce(MPI.IN_PLACE, p.grad, op=MPI.SUM)

            comm.Barrier()

            optimizer.step()

            if args.use_wandb and rank == 0:
                wandb.log(
                    {
                        "loss": loss.item(),
                        "accuracy": acc.item(),
                        "epoch": epoch,
                        "examples": epoch * batch_size,
                    }
                )

            if (epoch - 1) % args.print_interval == 0 and rank == 0:
                print(
                    f"Rank {rank} - Epoch {epoch} - Loss {loss.item()} - Accuracy {acc.item()}"
                )

            if (
                (epoch - 1) % args.save_interval == 0
                and rank == 0
                and args.save_path is not None
            ):
                torch.save(model.state_dict(), args.save_path)

    if args.use_wandb:
        wandb.finish()

    for h in hooks:
        h.remove()

    torch.save(model.state_dict(), args.save_path)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train a transformer language model, using MPI for distributed training."
    )
    parser.add_argument("--use_wandb", type=bool, default=False, help="Use wandb.")
    parser.add_argument("--d_model", type=int, required=True, help="Model dimension.")
    parser.add_argument(
        "--n_heads", type=int, required=True, help="Number of attention heads."
    )
    parser.add_argument("--n_layers", type=int, required=True, help="Number of layers.")
    parser.add_argument("--n_ctx", type=int, required=True, help="Context length.")
    parser.add_argument("--d_vocab", type=int, required=True, help="Vocabulary size.")
    parser.add_argument("--data", type=str, default=None, help="Path to the data file.")
    parser.add_argument(
        "--task", type=str, default="reverse_seq", help="Task to train on."
    )
    parser.add_argument(
        "--device_count", type=int, default=1, help="Number of devices to use."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.02, help="Weight decay."
    )
    parser.add_argument(
        "--d_vocab_out", type=int, default=None, help="Output Vocabulary size."
    )
    parser.add_argument("--use_mlp", type=bool, default=True, help="Use MLPs.")
    parser.add_argument("--d_mlp", type=int, default=None, help="MLP dimension.")
    parser.add_argument(
        "--use_norm", type=bool, default=True, help="Use layer normalization."
    )
    parser.add_argument(
        "--eps", type=float, default=1e-5, help="Normalization epsilon."
    )
    parser.add_argument(
        "--act_fn", type=str, default="gelu", help="Activation function."
    )
    parser.add_argument(
        "--weight_std",
        type=float,
        default=0.02,
        help="Weight initialization standard deviation.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--print_interval", type=int, default=100, help="Print interval.")
    parser.add_argument(
        "--save_interval", type=int, default=1000, help="Save interval."
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="Path to save the model."
    )

    args = parser.parse_args()

    print(f"Args parsed: {args}.")

    if args.device_count > 1:
        assert args.device == "cuda", "Distributed training is only supported on GPU."
        assert (
            args.device_count <= torch.cuda.device_count()
        ), "Not enough GPUs available."

    assert args.act_fn in [
        "relu",
        "gelu",
    ], "Activation function must be either relu or gelu."

    train(args)