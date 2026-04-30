import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.exposure import match_histograms


class ColorMLP(nn.Module):
    def __init__(self, hidden=32, strength=0.35):
        super().__init__()
        self.strength = strength
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.strength * self.net(x)


def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img).astype(np.float32) / 255.0


def resize_to_ref(img, ref):
    return np.array(
        Image.fromarray((img * 255).astype(np.uint8))
        .resize((ref.shape[1], ref.shape[0]), Image.BILINEAR)
    ).astype(np.float32) / 255.0


def project_orthogonal_(M):
    with torch.no_grad():
        U, _, Vt = torch.linalg.svd(M)
        R = U @ Vt
        if torch.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        M.copy_(R)


def harmonize(
    ref_path,
    tar_path,
    iterations=1000,
    max_pixels=200_000,
    mapping="ortho",
    hist=False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img1 = load_image(ref_path)
    img2 = load_image(tar_path)

    img1 = resize_to_ref(img1, img2)

    A = torch.tensor(img1, dtype=torch.float32, device=device)
    B = torch.tensor(img2, dtype=torch.float32, device=device)

    mask = (
        (A.mean(dim=2) < 0.97) &
        (B.mean(dim=2) < 0.97) &
        (A.mean(dim=2) > 0.03) &
        (B.mean(dim=2) > 0.03)
    )

    A_pix = A[mask]
    B_pix = B[mask]

    n = min(A_pix.shape[0], max_pixels)
    idx = torch.randperm(A_pix.shape[0], device=device)[:n]
    A_pix = A_pix[idx]
    B_pix = B_pix[idx]

    if mapping in {"unconst", "ortho"}:
        M = nn.Parameter(torch.eye(3, device=device))
        opt = optim.Adam([M], lr=1e-2)

        for _ in tqdm(range(iterations), desc=f"Optimizing {mapping} transform"):
            opt.zero_grad()

            pred = B_pix @ M
            loss = torch.sqrt(((pred - A_pix) ** 2).mean())

            if mapping == "unconst":
                reg = 1e-6 * ((M.T @ M - torch.eye(3, device=device)) ** 2).mean()
                total = loss + reg
            else:
                total = loss

            total.backward()
            opt.step()

            if mapping == "ortho":
                project_orthogonal_(M)

        with torch.no_grad():
            img2_post = B.reshape(-1, 3) @ M
            img2_post = torch.clamp(img2_post, 0, 1)
            img2_post = img2_post.reshape(B.shape).cpu().numpy()

        print("M:")
        print(M.detach().cpu().numpy())

    elif mapping == "MLP":
        model = ColorMLP(hidden=32, strength=0.35).to(device)
        opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        for _ in tqdm(range(iterations), desc="Optimizing MLP transform"):
            opt.zero_grad()

            pred = model(B_pix)
            loss = torch.sqrt(((pred - A_pix) ** 2).mean())

            range_penalty = (
                torch.relu(-pred).pow(2).mean()
                + torch.relu(pred - 1).pow(2).mean()
            )

            total = loss + 0.1 * range_penalty
            total.backward()
            opt.step()

        with torch.no_grad():
            img2_post = model(B.reshape(-1, 3))
            img2_post = torch.clamp(img2_post, 0, 1)
            img2_post = img2_post.reshape(B.shape).cpu().numpy()

    else:
        raise ValueError(f"Unknown mapping: {mapping}")

    if hist==1:
        img2_post = match_histograms(img2_post, img1, channel_axis=-1)
        img2_post = np.clip(img2_post, 0, 1)

    out_base = os.path.splitext(tar_path)[0]
    hist_suffix = "_hist" if hist else ""
    out_path = f"{out_base}_harmonized_{mapping}{hist_suffix}.png"

    Image.fromarray((img2_post * 255).astype(np.uint8)).save(out_path)
    print(f"Saved: {out_path}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img1)
    axes[0].set_title("reference")

    axes[1].imshow(img2)
    axes[1].set_title("target pre")

    axes[2].imshow(img2_post)
    axes[2].set_title(f"target post ({mapping}{' + hist' if hist else ''})")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    return img2_post


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ref", required=True, help="reference image")
    parser.add_argument("-tar", required=True, help="target image")
    parser.add_argument("-ite", type=int, default=1000, help="number of iterations")
    parser.add_argument(
        "-mapping",
        choices=["unconst", "ortho", "MLP"],
        default="ortho",
        help="color mapping type",
    )
    parser.add_argument(
        "-hist",
        type=int,
        default=1,
        help="apply histogram matching (1=on, 0=off, default=1)",
    )

    args = parser.parse_args()

    harmonize(
        ref_path=args.ref,
        tar_path=args.tar,
        iterations=args.ite,
        mapping=args.mapping,
        hist=args.hist,
    )


if __name__ == "__main__":
    main()
