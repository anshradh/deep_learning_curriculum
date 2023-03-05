import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import torchvision
from torch.utils.data import DataLoader, Subset
from model import MNISTCNN, MNISTCNNConfig
from argparse import ArgumentParser, Namespace
import math
import torch.distributed as dist
import os
import torch.multiprocessing as mp


def train(args: Namespace):
    rank = dist.get_rank() if dist.is_initialized() else 0
    size = dist.get_world_size() if dist.is_initialized() else 1

    device = torch.device(args.device, rank)

    config = MNISTCNNConfig(
        d_kernel=args.d_kernel,
        n_filters=args.n_filters,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )

    lr_correction_factor = math.sqrt(32) / math.sqrt(args.n_filters)

    start_seed = args.start_seed + rank

    final_val_losses = []

    for s in range(start_seed, start_seed + args.num_seeds):
        torch.manual_seed(s)

        model = MNISTCNN(config).to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr_correction_factor * args.learning_rate,
            weight_decay=args.weight_decay,
        )

        train_dataset = torchvision.datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )

        train_dataset = Subset(
            train_dataset,
            torch.randperm(int(len(train_dataset) * args.dataset_fraction)).tolist(),
        )

        val_dataset = torchvision.datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )

        local_train_dataset = Subset(
            train_dataset,
            range(
                rank * len(train_dataset) // size,
                (rank + 1) * len(train_dataset) // size,
            ),
        )

        local_val_dataset = Subset(
            val_dataset,
            range(
                rank * len(val_dataset) // size, (rank + 1) * len(val_dataset) // size
            ),
        )

        train_loader = DataLoader(
            local_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
        )

        val_loader = DataLoader(
            local_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

        model.train()

        hooks = []
        if dist.is_initialized():
            for p in model.parameters():
                dist.broadcast(p.data, 0)
                hooks.append(
                    p.register_hook(lambda grad: grad / size if grad is not None else grad)
                )

        if args.use_wandb:
            import wandb

            wandb.init(project=args.wandb_project, config=args)

        for epoch in tqdm(range(1, args.epochs + 1)):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                accuracy = (output.argmax(dim=1) == target).float().mean()
                loss.backward()

                if dist.is_initialized():
                    for p in model.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)


                optimizer.step()

                if args.use_wandb and rank == 0:
                    wandb.log(
                        {
                            "train_loss": loss.item(),
                            "train_accuracy": accuracy.item(),
                        }
                    )

                if (
                    args.print_interval is not None
                    and batch_idx % args.print_interval == 0
                ):
                    print(
                        f"Train Epoch: {epoch} \tLoss: {loss.item():.6f}\tAccuracy: {accuracy.item():.6f}"
                    )

                if (
                    args.save_interval is not None
                    and batch_idx % args.save_interval == 0
                    and rank == 0
                ):
                    torch.save(
                        model.state_dict(),
                        f"{args.model_save_path}mnist_model_{epoch}_{batch_idx}.pt",
                    )

            with torch.no_grad():
                model.eval()
                val_loss = 0.
                val_accuracy = 0.
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += F.cross_entropy(output, target, reduction="sum")
                    val_accuracy += (
                        (output.argmax(dim=1) == target).float().sum()
                    )
                val_loss /= len(local_val_dataset)
                val_accuracy /= len(local_val_dataset)

                val_loss = dist.all_reduce(val_loss / size, op=dist.ReduceOp.SUM)
                val_accuracy = dist.all_reduce(val_accuracy / size, op=dist.ReduceOp.SUM)

                val_loss = val_loss.item()
                val_accuracy = val_accuracy.item()


                if rank == 0:
                    print(
                        f"Val set: Average loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}"
                    )
                model.train()
                if args.use_wandb and rank == 0:
                    wandb.log(
                        {
                            "val_loss": val_loss,
                            "val_accuracy": val_accuracy,
                        }
                    )
                if epoch == args.epochs and rank == 0:
                    final_val_losses.append(val_loss)

        for hook in hooks:
            hook.remove()

        if rank == 0:
            torch.save(
                model.state_dict(),
                f"{args.model_save_path}mnist_model_final_n_filters_{config.n_filters}_dataset_frac_{args.dataset_fraction:3f}_seed_{s}.pt",
            )

        if args.use_wandb and rank == 0:
            wandb.finish()

    avg_val_loss = sum(final_val_losses) / len(final_val_losses)
    if rank == 0:
        print(f"Average val loss: {avg_val_loss}")

    if rank == 0:
        torch.save(
            dict(
                loss=avg_val_loss,
                n_filters=config.n_filters,
                dataset_fraction=args.dataset_fraction,
            ),
            f"{args.results_save_path}mnist_model_final_n_filters_{config.n_filters}_dataset_frac_{args.dataset_fraction:3f}_results.pt",
        )

def rank_process(rank, world_size, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    store = dist.TCPStore("127.0.0.1", 29500, world_size, rank == 0)
    dist.init_process_group("nccl", store=store, rank=rank, world_size=world_size,)
    train(args)

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--device", type=str, default="cpu")
    arg_parser.add_argument("--device_count", type=int, default=1)
    arg_parser.add_argument("--start_seed", type=int, default=0)
    arg_parser.add_argument("--num_seeds", type=int, default=1)
    arg_parser.add_argument("--batch_size", type=int, default=64)
    arg_parser.add_argument("--learning_rate", type=float, default=1e-3)
    arg_parser.add_argument("--dataset_fraction", type=float, default=1.0)
    arg_parser.add_argument("--epochs", type=int, default=10)
    arg_parser.add_argument("--d_kernel", type=int, default=3)
    arg_parser.add_argument("--n_filters", type=int, default=32)
    arg_parser.add_argument("--n_layers", type=int, default=5)
    arg_parser.add_argument("--dropout", type=float, default=0.25)
    arg_parser.add_argument("--weight_decay", type=float, default=0.02)
    arg_parser.add_argument("--use_wandb", type=bool, default=False)
    arg_parser.add_argument("--wandb_project", type=str, default="scaling_laws")
    arg_parser.add_argument("--print_interval", type=int, default=None)
    arg_parser.add_argument("--save_interval", type=int, default=None)
    arg_parser.add_argument("--model_save_path", type=str, default="")
    arg_parser.add_argument("--results_save_path", type=str, required=True)

    args = arg_parser.parse_args()

    print(f"Args parsed: {args}")

    if args.device_count > 1:
        assert args.device == "cuda", "Distributed training is only supported on GPU."
        assert (
            args.device_count <= torch.cuda.device_count()
        ), "Not enough GPUs available."

    if args.device_count > 1:
        mp.spawn(
            rank_process,
            args=(args.device_count, args),
            nprocs=args.device_count,
            join=True,
        )
    else:
        train(args)
