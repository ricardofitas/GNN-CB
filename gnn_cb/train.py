"""Reproducible two-specimen, leave-one-specimen-out pilot benchmark.

Run ``python -m gnn_cb.train --epochs 600 --seeds 11 29 47``.
All fitting and epoch selection use only the training specimen. A contiguous
late-time validation block is separated from fitting frames by a phase gap.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .data import load_dataset
from .model import DeformationModel, ModelConfig, affine_features


def tensor_data(data: dict) -> dict:
    return {key: torch.as_tensor(data[key], dtype=(torch.long if key == "edge_index" else
                 torch.bool if key == "mask" else torch.float32))
            for key in ("x0", "y", "mask", "edge_index", "progress")}


def partition(progress: np.ndarray) -> dict[str, np.ndarray]:
    """Deterministic chronological split, with no frame-level randomization."""
    valid = (progress >= 0) & (progress <= 1)
    return {"train": valid & (progress <= 0.75),
            "validation": valid & (progress >= 0.80),
            "gap": valid & (progress > 0.75) & (progress < 0.80),
            "test": valid}


def fit_affine(data: dict, selected: np.ndarray) -> torch.Tensor:
    """Fit quadratic protocol phase x affine initial-coordinate displacement."""
    x0 = torch.as_tensor(data["x0"], dtype=torch.float64)
    progress = torch.as_tensor(data["progress"][selected], dtype=torch.float64)
    features = affine_features(x0, progress).numpy()
    mask = data["mask"][selected]
    coefficients, *_ = np.linalg.lstsq(features[mask], data["y"][selected][mask], rcond=1e-10)
    return torch.tensor(coefficients, dtype=torch.float32)


def metrics(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray, scale: float) -> dict:
    prediction, truth = prediction[mask], truth[mask]
    error = prediction - truth
    mse = float(np.mean(error ** 2))
    epe = np.linalg.norm(error, axis=-1)
    sse = float(np.sum(error ** 2))
    sst = float(np.sum((truth - truth.mean(axis=0, keepdims=True)) ** 2))
    return {"observed_node_frames": int(len(error)),
            "rmse_normalized": float(np.sqrt(mse)),
            "rmse_px": float(np.sqrt(mse) * scale),
            "mae_px": float(np.mean(np.abs(error)) * scale),
            "mean_endpoint_error_px": float(epe.mean() * scale),
            "mean_endpoint_error_normalized": float(epe.mean()),
            "p95_endpoint_error_px": float(np.quantile(epe, 0.95) * scale),
            "r2_variance_weighted": 1.0 - sse / sst if sst > 0 else None,
            "rmse_x_px": float(np.sqrt(np.mean(error[:, 0] ** 2)) * scale),
            "rmse_y_px": float(np.sqrt(np.mean(error[:, 1] ** 2)) * scale)}


def train_one(data: dict, kind: str, seed: int, epochs: int,
              patience: int, learning_rate: float) -> tuple[DeformationModel, list[dict], dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    split = partition(data["progress"])
    tensors = tensor_data(data)
    coefficients = fit_affine(data, split["train"])
    train_indices = torch.from_numpy(np.flatnonzero(split["train"]))
    val_indices = torch.from_numpy(np.flatnonzero(split["validation"]))
    x0, edges = tensors["x0"], tensors["edge_index"]
    train_p, train_y, train_mask = (tensors[key][train_indices]
                                  for key in ("progress", "y", "mask"))
    val_p, val_y, val_mask = (tensors[key][val_indices]
                            for key in ("progress", "y", "mask"))
    base_train = affine_features(x0, train_p) @ coefficients
    residual_scale = max(float(((base_train - train_y)[train_mask].square().mean()).sqrt()), 0.01)
    model = DeformationModel(ModelConfig(kind=kind), coefficients, residual_scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    history, best_value, best_epoch = [], float("inf"), 0
    best_state = copy.deepcopy(model.state_dict())
    start = time.monotonic()
    for epoch in range(epochs + 1):
        if epoch:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pred = model(x0, edges, train_p)
            loss = ((pred - train_y)[train_mask] / residual_scale).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                train_mse = float((model(x0, edges, train_p) - train_y)[train_mask].square().mean())
                val_mse = float((model(x0, edges, val_p) - val_y)[val_mask].square().mean())
            history.append({"epoch": epoch, "train_mse_normalized": train_mse,
                            "validation_mse_normalized": val_mse})
            if val_mse < best_value - 1e-10:
                best_value, best_epoch = val_mse, epoch
                best_state = copy.deepcopy(model.state_dict())
            if epoch - best_epoch >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    info = {"seed": seed, "selected_epoch": best_epoch, "completed_epochs": epoch,
            "selected_validation_mse_normalized": best_value,
            "runtime_seconds": time.monotonic() - start,
            "residual_scale": residual_scale,
            "fit_frames": int(split["train"].sum()),
            "validation_frames": int(split["validation"].sum()),
            "purged_frames": int(split["gap"].sum()), **model.metadata()}
    return model, history, info


def write_prediction(path: Path, data: dict, pred: np.ndarray, selection: np.ndarray) -> None:
    """Long CSV includes missingness explicitly; missing targets have blank values."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["specimen", "frame", "phase", "node_id", "observed",
                         "initial_x_px", "initial_y_px", "true_dx_px", "true_dy_px",
                         "pred_dx_px", "pred_dy_px", "endpoint_error_px"])
        scale = float(data["scale_px"])
        for i in np.flatnonzero(selection):
            for j, node in enumerate(data["node_ids"]):
                observed = bool(data["mask"][i, j])
                true = data["y"][i, j] * scale
                predicted = pred[i, j] * scale
                writer.writerow([str(data["specimen_id"]), int(data["frame"][i]),
                                 float(data["progress"][i]), int(node), int(observed),
                                 *map(float, data["initial_pos_px"][j]),
                                 *(map(float, true) if observed else ("", "")),
                                 *map(float, predicted),
                                 float(np.linalg.norm(true - predicted)) if observed else ""])


def evaluate(data: dict, pred: np.ndarray) -> dict:
    p = data["progress"]
    selections = {"compression_window": (p >= 0) & (p <= 1),
                  "within_fit_phase_support": (p >= 0) & (p <= 0.75),
                  "late_phase": (p >= 0.80) & (p <= 1)}
    result = {name: metrics(pred[selection], data["y"][selection],
                            data["mask"][selection], float(data["scale_px"]))
              for name, selection in selections.items()}
    window = selections["compression_window"]
    complete = data["mask"][window].all(axis=0)
    stable_mask = data["mask"][window] & complete[None, :]
    result["complete_track_subset"] = {
        "nodes": int(complete.sum()),
        "node_ids": data["node_ids"][complete].astype(int).tolist(),
        **metrics(pred[window], data["y"][window], stable_mask, float(data["scale_px"]))}
    result["framewise"] = [
        {"frame": int(data["frame"][i]), "phase": float(p[i]),
         "observed_nodes": int(data["mask"][i].sum()),
         "rmse_px": metrics(pred[i], data["y"][i], data["mask"][i],
                            float(data["scale_px"]))["rmse_px"]}
        for i in np.flatnonzero(window)]
    return result


def fit_final(datasets: list[dict], summary: dict, output: Path,
              learning_rate: float = 0.002, seed: int = 11) -> dict:
    """Fit a demonstration model to BOTH specimens; no independent test claim.

    Epoch budget is the rounded median of source-only GNN validation selections,
    fixed before this joint fit. All available compression phases are then used.
    """
    output.mkdir(parents=True, exist_ok=True)
    epochs = int(round(float(np.median([m["selected_epoch"] for f in summary["folds"]
                                       for m in f["models"]["gnn"]]))))
    torch.manual_seed(seed)
    np.random.seed(seed)
    tensors = [tensor_data(d) for d in datasets]
    selections = [partition(d["progress"])["test"] for d in datasets]
    features, targets = [], []
    for d, t, selected in zip(datasets, tensors, selections):
        mask = d["mask"][selected]
        features.append(affine_features(t["x0"].double(), t["progress"][selected].double()).numpy()[mask])
        targets.append(d["y"][selected][mask])
    x, y = np.concatenate(features), np.concatenate(targets)
    coeff = torch.tensor(np.linalg.lstsq(x, y, rcond=1e-10)[0], dtype=torch.float32)
    residual_scale = max(float(np.sqrt(np.mean((x @ coeff.numpy() - y) ** 2))), 0.01)
    model = DeformationModel(ModelConfig(kind="gnn"), coeff, residual_scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    history = []
    for epoch in range(epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for t, selected in zip(tensors, selections):
            mask = t["mask"][selected]
            pred = model(t["x0"], t["edge_index"], t["progress"][selected])
            losses.append(((pred[mask] - t["y"][selected][mask]) / residual_scale).square().mean())
        loss = torch.stack(losses).mean()
        if epoch:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                post_losses = []
                for t, selected in zip(tensors, selections):
                    mask = t["mask"][selected]
                    pred = model(t["x0"], t["edge_index"], t["progress"][selected])
                    post_losses.append((pred[mask] - t["y"][selected][mask]).square().mean())
            history.append({"epoch": epoch, "train_mse_normalized": float(torch.stack(post_losses).mean())})
    model.eval()
    metadata = {"config": asdict(model.config), "training_specimens": [str(d["specimen_id"]) for d in datasets],
                "seed": seed, "selected_epoch": epochs,
                "epoch_rule": "rounded median GNN source-only validation-selected epochs in LOSO benchmark",
                "input_semantics": summary["task"],
                "normalization": "initial graph height; centered initial xy neural features",
                "fit_phase": [0, 1], "independent_evaluation": False,
                "warning": "Both available independent specimens used for training. Demonstration model only.",
                **model.metadata()}
    torch.save({"state_dict": model.state_dict(), **metadata}, output / "model.pt")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return metadata


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    datasets = load_dataset(args.data)
    if len(datasets) != 2:
        raise ValueError("This registered pilot protocol expects exactly two independent specimens.")
    summary = {"task": "retrospective protocol-phase-conditioned node displacement",
               "protocol": {"independent_specimens": 2, "outer_split": "leave-one-specimen-out",
                            "fit_phase": [0, 0.75], "purge_phase_open": [0.75, 0.80],
                            "validation_phase": [0.80, 1.0], "test_phase": [0, 1],
                            "epoch_selection": "training specimen validation MSE only; includes epoch zero",
                            "seeds": args.seeds, "maximum_epochs": args.epochs,
                            "patience_epochs": args.patience, "learning_rate": args.learning_rate,
                            "weight_decay": 0.001, "loss": "masked coordinate MSE",
                            "test_information_used_for_training": False},
               "environment": {"torch": torch.__version__, "numpy": np.__version__,
                               "device": "cpu", "threads": args.threads}, "folds": []}
    for train, test in ((datasets[0], datasets[1]), (datasets[1], datasets[0])):
        train_id, test_id = str(train["specimen_id"]), str(test["specimen_id"])
        fold_dir = output / f"train_{train_id}__test_{test_id}"
        fold_dir.mkdir(exist_ok=True)
        split = partition(train["progress"])
        coefficients = fit_affine(train, split["train"])
        test_t = tensor_data(test)
        with torch.no_grad():
            affine = (affine_features(test_t["x0"], test_t["progress"]) @ coefficients).numpy()
        fold = {"training_specimen": train_id, "held_out_specimen": test_id,
                "training_frame_ids": train["frame"][split["train"]].astype(int).tolist(),
                "validation_frame_ids": train["frame"][split["validation"]].astype(int).tolist(),
                "purged_frame_ids": train["frame"][split["gap"]].astype(int).tolist(),
                "baselines": {"zero_displacement": evaluate(test, np.zeros_like(test["y"])),
                              "phase_affine": evaluate(test, affine)}, "models": {}}
        write_prediction(fold_dir / "predictions_phase_affine.csv", test, affine, partition(test["progress"])["test"])
        for kind in ("gnn", "mlp"):
            fold["models"][kind] = []
            for seed in args.seeds:
                model, history, info = train_one(train, kind, seed, args.epochs,
                                                 args.patience, args.learning_rate)
                with torch.no_grad():
                    pred = model(test_t["x0"], test_t["edge_index"], test_t["progress"]).numpy()
                result = {**info, "test": evaluate(test, pred)}
                stem = f"{kind}_seed{seed}"
                checkpoint = {"state_dict": model.state_dict(), "config": asdict(model.config),
                              "training_specimen": train_id, "held_out_specimen": test_id,
                              "seed": seed, "selected_epoch": info["selected_epoch"],
                              "input_semantics": summary["task"],
                              "normalization": "initial graph height; centered initial xy neural features",
                              "protocol": summary["protocol"]}
                torch.save(checkpoint, fold_dir / f"{stem}.pt")
                (fold_dir / f"{stem}_history.json").write_text(json.dumps(history, indent=2) + "\n")
                write_prediction(fold_dir / f"{stem}_predictions.csv", test, pred, partition(test["progress"])["test"])
                np.savez_compressed(fold_dir / f"{stem}_predictions.npz", prediction=pred,
                                    frame=test["frame"], node_ids=test["node_ids"], mask=test["mask"])
                fold["models"][kind].append(result)
                print(json.dumps({"train": train_id, "test": test_id, "model": kind,
                                  "seed": seed, "epoch": info["selected_epoch"],
                                  "test_rmse_px": result["test"]["compression_window"]["rmse_px"]}), flush=True)
        summary["folds"].append(fold)
        (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Seeds measure optimizer variability, never independent experimental replication.
    aggregate = {}
    for kind in ("gnn", "mlp"):
        by_seed = [np.mean([fold["models"][kind][j]["test"]["compression_window"]["rmse_px"]
                           for fold in summary["folds"]]) for j in range(len(args.seeds))]
        aggregate[kind] = {"macro_rmse_px_mean_over_seeds": float(np.mean(by_seed)),
                           "macro_rmse_px_sd_over_seeds": float(np.std(by_seed, ddof=1)) if len(by_seed) > 1 else 0,
                           "note": "seed variability only; no population confidence interval"}
    for kind in ("zero_displacement", "phase_affine"):
        aggregate[kind] = {"macro_rmse_px": float(np.mean([f["baselines"][kind]["compression_window"]["rmse_px"] for f in summary["folds"]]))}
    summary["aggregate"] = aggregate
    summary["final_demonstration_model"] = fit_final(datasets, summary, output.parent / "final", args.learning_rate)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--output", default="results/loso")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
