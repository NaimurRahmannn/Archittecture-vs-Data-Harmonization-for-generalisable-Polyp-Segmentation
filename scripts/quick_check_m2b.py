import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_common import ManifestSegDataset, build_model_from_config, filter_manifest, load_simple_yaml, read_manifest, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Quick shape and gate checks for M2b.")
    parser.add_argument("--config", default="configs/m2b_convnext_xattn.yaml")
    parser.add_argument("--batch_size", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    set_seed(cfg.get("seed", 0))

    rows = read_manifest(cfg["manifest_csv"])
    train_rows = filter_manifest(rows, split="train")
    if not train_rows:
        raise SystemExit("No train rows found in manifest.")

    dataset = ManifestSegDataset(
        train_rows,
        cfg["data_root"],
        image_size=cfg.get("image_size", 256),
        augment=False,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    images, _ = next(iter(loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"model={model.__class__.__name__}")
    print(f"total_params={total_params}")
    print(f"input_shape={tuple(images.shape)}")

    shape_logs = []
    hooks = []
    for module_name, module in model.named_modules():
        if module.__class__.__name__ == "CrossAttentionPreserveFusion":
            def _make_hook(name):
                def _hook(mod, inputs, output):
                    dec_feat, enc_feat = inputs
                    shape_logs.append(
                        (
                            name,
                            tuple(dec_feat.shape),
                            tuple(enc_feat.shape),
                            tuple(output.shape),
                            dec_feat.shape[-2:] == enc_feat.shape[-2:],
                        )
                    )
                return _hook
            hooks.append(module.register_forward_hook(_make_hook(module_name)))

    with torch.no_grad():
        logits = model(images.to(device))
    print(f"logits_shape={tuple(logits.shape)}")

    for hook in hooks:
        hook.remove()

    print("fusion_shape_checks:")
    for name, dec_shape, skip_shape, out_shape, same_hw in shape_logs:
        print(
            f"{name}: dec={dec_shape} skip={skip_shape} out={out_shape} "
            f"same_hw={'YES' if same_hw else 'NO'}"
        )

    print("gate_values:")
    for module_name, module in model.named_modules():
        if hasattr(module, "alpha_logit") and hasattr(module, "beta_logit"):
            alpha = torch.sigmoid(module.alpha_logit.detach()).item()
            beta = torch.sigmoid(module.beta_logit.detach()).item()
            print(f"{module_name}: alpha={alpha:.4f} beta={beta:.4f}")


if __name__ == "__main__":
    main()
