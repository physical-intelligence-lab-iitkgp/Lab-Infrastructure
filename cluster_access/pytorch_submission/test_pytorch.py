import os
import time
import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 1024, hidden_dim: int = 512, out_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"PyTorch version : {torch.__version__}")
    print(f"Using device    : {device}")
    if device.type == "cuda":
        print(f"GPU count       : {torch.cuda.device_count()}")
        print(f"GPU name (0)    : {torch.cuda.get_device_name(0)}")

    model = TinyMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(256, 1024, device=device)
    targets = torch.randint(0, 10, (256,), device=device)

    model.train()
    for step in range(1, 21):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        if step % 5 == 0 or step == 1:
            print(f"Step {step:02d}/20 | Loss: {loss.item():.4f}")

    if device.type == "cuda":
        max_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Peak GPU memory : {max_mem_mb:.2f} MB")

    elapsed = time.time() - start_time
    print(f"Elapsed time    : {elapsed:.2f}s")
    print("PyTorch submission test PASSED.")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()