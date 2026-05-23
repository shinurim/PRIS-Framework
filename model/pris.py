import math
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Proposed:
    def __init__(self, **args_config):
        self.args_config = dict(args_config)

    def PRIS(self, emb_dim: int, hc_dim: int, meta_dim: int) -> nn.Module:
        dropout_rate = self.args_config.get("dropout_rate", 0.2)

        # ===== 1. Pyramid MLP Block (Linear → BN → GELU → Dropout + Residual) =====
        class PyramidBlock(nn.Module):
            def __init__(self, in_dim, out_dim, dropout):
                super().__init__()
                self.fc = nn.Linear(in_dim, out_dim)
                self.bn = nn.BatchNorm1d(out_dim)
                self.act = nn.GELU()
                self.drop = nn.Dropout(dropout)
                self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

            def forward(self, x):
                out = self.drop(self.act(self.bn(self.fc(x))))
                return out + self.skip(x)

        # ===== 2. Element-wise Fusion + Pyramid MLP + Head =====
        input_dim = emb_dim * 4 + hc_dim + meta_dim
        mlp = nn.Sequential(
            PyramidBlock(input_dim, 1024, dropout_rate),
            PyramidBlock(1024, 512, dropout_rate),
            PyramidBlock(512, 256, dropout_rate),
        )
        head = nn.Linear(256, 1)

        class ReviewPatchRHP(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = mlp
                self.head = head

            def forward(self, r_cls, p_cls, hc, meta):
                x = torch.cat([
                    r_cls, p_cls,
                    r_cls * p_cls,
                    r_cls - p_cls,
                    hc,
                    meta,
                ], dim=-1)
                x = self.mlp(x)
                return self.head(x).squeeze(-1)

        return ReviewPatchRHP().to(DEVICE)


def model_compile(model, args):
    lr = args.get("learning_rate", 3e-4)
    wd = args.get("weight_decay", 1e-4)
    epochs = args.get("epochs", 100)
    beta = args.get("loss_beta", 0.5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.SmoothL1Loss(beta=beta)

    return model, optimizer, scheduler, loss_fn


def train_model(model, train_r, train_p, train_hc, train_meta, train_y,
                val_r, val_p, val_hc, val_meta, val_y, args, save_path: str = None):
    batch_size = args.get("batch_size", 128)
    epochs     = args.get("epochs", 100)
    patience   = args.get("patience", 10)

    use_amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16
    torch.backends.cudnn.benchmark = True

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_r), torch.from_numpy(train_p),
                      torch.from_numpy(train_hc), torch.from_numpy(train_meta),
                      torch.from_numpy(train_y)),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_r), torch.from_numpy(val_p),
                      torch.from_numpy(val_hc), torch.from_numpy(val_meta),
                      torch.from_numpy(val_y)),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    model, optimizer, scheduler, loss_fn = model_compile(model, args)

    def run_epoch(loader, train: bool):
        model.train(train)
        tot_loss = torch.zeros(1, device=DEVICE)
        n_total = 0
        preds_list, tgts_list = [], []

        ctx = torch.enable_grad() if train else torch.inference_mode()
        with ctx:
            for r_cls, p_cls, hc, meta, y in loader:
                r_cls = r_cls.to(DEVICE, non_blocking=True)
                p_cls = p_cls.to(DEVICE, non_blocking=True)
                hc    = hc.to(DEVICE, non_blocking=True)
                meta  = meta.to(DEVICE, non_blocking=True)
                y     = y.to(DEVICE, non_blocking=True)

                with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    pred = model(r_cls, p_cls, hc, meta)
                    loss = loss_fn(pred.float(), y)

                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                bs = y.size(0)
                tot_loss += loss.detach() * bs
                n_total  += bs
                preds_list.append(pred.detach().float())
                tgts_list.append(y.detach())

        preds = torch.cat(preds_list).cpu().numpy()
        tgts  = torch.cat(tgts_list).cpu().numpy()
        avg_loss = (tot_loss / n_total).item()
        mse  = float(mean_squared_error(tgts, preds))
        rmse = math.sqrt(mse)
        mae  = float(mean_absolute_error(tgts, preds))
        rho  = spearmanr(tgts, preds).correlation if len(tgts) > 1 else 0.0
        return avg_loss, rmse, mae, rho

    best_val = float("inf")
    best_state = None
    bad = 0
    history = {"train_loss": [], "val_loss": []}

    for ep in range(1, epochs + 1):
        tr_loss, _, _, _ = run_epoch(train_loader, train=True)
        va_loss, va_rmse, va_mae, va_rho = run_epoch(val_loader, train=False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        if ep % 5 == 0 or ep == 1:
            print(f"[ep {ep:3d}] tr_loss={tr_loss:.4f}  va_loss={va_loss:.4f}  "
                  f"rmse={va_rmse:.4f}  mae={va_mae:.4f}  rho={va_rho:.4f}")

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n학습 완료: best val_loss={best_val:.4f}")

    # best 가중치를 .pt로 저장 (state_dict만). save_path가 주어졌을 때만.
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        state_to_save = best_state if best_state is not None else model.state_dict()
        torch.save(state_to_save, save_path)
        print(f"[checkpoint] best weights saved → {save_path}")

    return history


def predict(model, test_r, test_p, test_hc, test_meta, batch_size: int = 128):
    use_amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16

    loader = DataLoader(
        TensorDataset(torch.from_numpy(test_r), torch.from_numpy(test_p),
                      torch.from_numpy(test_hc), torch.from_numpy(test_meta)),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    model.eval()
    preds_list = []
    with torch.inference_mode():
        for r_cls, p_cls, hc, meta in loader:
            r_cls = r_cls.to(DEVICE, non_blocking=True)
            p_cls = p_cls.to(DEVICE, non_blocking=True)
            hc    = hc.to(DEVICE, non_blocking=True)
            meta  = meta.to(DEVICE, non_blocking=True)
            with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                pred = model(r_cls, p_cls, hc, meta)
            preds_list.append(pred.detach().float().cpu())
    return torch.cat(preds_list).numpy()
