"""
train.py
────────────────────────────────────────────────────────────────────────────────
Three-modality training: Text (BERT) + Video (ResNet-18) + Audio (MFCC MLP)
Fusion: FusionModelV3 (cross-modal attention)
────────────────────────────────────────────────────────────────────────────────
"""

import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

from dataset                    import MultiDataset
from models.text_model          import TextModel
from models.video_model         import VideoModel
from models.attention_fusion_v3 import FusionModelV3


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using: {device}")

    full  = MultiDataset("dataset.json")
    test_n = int(0.15 * len(full))
    val_n  = int(0.15 * len(full))
    train_n = len(full) - val_n - test_n
    train_ds, val_ds, test_ds = random_split(
        full,
        [train_n, val_n, test_n],
        generator=torch.Generator().manual_seed(42)
)

    # ── Class-balanced sampling (avoids "always FAKE" bias) ───────────────────
    labels       = [full[i]["label"].item() for i in train_ds.indices]
    class_counts = [labels.count(0), labels.count(1)]
    print(f"Train labels → real: {class_counts[0]}, fake: {class_counts[1]}")
    weights = [1.0 / class_counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=4, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False,   num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    # ── Models ────────────────────────────────────────────────────────────────
    tm = TextModel().to(device)
    vm = VideoModel().to(device)
    fm = FusionModelV3().to(device)

    trainable = (
        list(tm.proj.parameters()) +
        list(filter(lambda p: p.requires_grad, tm.bert.parameters())) +
        list(vm.proj.parameters()) +
        list(fm.parameters())
    )

    opt       = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
    loss_fn   = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Training loop ─────────────────────────────────────────────────────────
    def run_epoch(loader, train=True):
        tm.train(train); vm.train(train); fm.train(train)
        total_loss, correct, total = 0, 0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for b in loader:
                i_ids = b["input_ids"].to(device)
                mask  = b["attention_mask"].to(device)
                v     = b["video"].to(device)
                y     = b["label"].to(device)
                batch_size = v.size(0)
                a_out = torch.zeros((batch_size, 256)).to(device)

                t_out  = tm(i_ids, mask)
                v_out  = vm(v)
                out    = fm(t_out, v_out, a_out)

                loss = loss_fn(out, y)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    opt.step()

                total_loss += loss.item()
                correct    += (out.argmax(1) == y).sum().item()
                total      += y.size(0)

        return total_loss / len(loader), correct / total

    best_val_acc = 0
    for epoch in range(5):
        tr_loss, tr_acc = run_epoch(train_loader, train=True)
        vl_loss, vl_acc = run_epoch(val_loader,   train=False)
        scheduler.step()
        print(
            f"Epoch {epoch+1} | "
            f"Train {tr_loss:.3f} / {tr_acc:.2%} | "
            f"Val   {vl_loss:.3f} / {vl_acc:.2%}"
        )
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                "tm": tm.state_dict(),
                "vm": vm.state_dict(),
                "fm": fm.state_dict(),
            }, "best_model.pth")
            print("  ✓ saved best model")

    print(f"\nBest val accuracy: {best_val_acc:.2%}")
    test_loss, test_acc = run_epoch(test_loader, train=False)
    print(f"\nTest Accuracy: {test_acc:.2%}")


if __name__ == "__main__":
    main()