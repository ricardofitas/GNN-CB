"""Specimen-separated benchmark for an expanded image-derived graph dataset.

The protocol is saved before fitting. Each physical specimen is tested once;
the next physical specimen in sorted order supplies validation labels only.
All remaining physical specimens supply fitting labels. Replicate extractions
of a physical specimen remain together and share that specimen's loss weight.

Run ``python -m gnn_cb.train_expanded --data data/expanded --protocol-only``
to inspect the exact split before starting the benchmark.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .data import load_specimen
from .model import DeformationModel, ModelConfig, affine_features, load_checkpoint
from .train import metrics, tensor_data, write_prediction


TASK = "retrospective protocol-phase-conditioned image-tracked node displacement"


def physical_id(data: dict) -> str:
    """An extraction ID is not evidence for an independent physical test."""
    return str(data.get("physical_specimen_id", data["specimen_id"]))


def compression_selection(data: dict) -> np.ndarray:
    return (data["progress"] >= 0) & (data["progress"] <= 1)


def load_expanded(directories: list[str], specimens: list[str] | None = None) -> list[dict]:
    datasets = []
    for directory in directories:
        for path in sorted(Path(directory).glob("*.npz")):
            data = load_specimen(path)
            specimen = str(data["specimen_id"])
            if specimens is not None and specimen not in specimens:
                continue
            metadata_path = path.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            validate_admission(metadata, specimen)
            if "physical_specimen_id" not in data:
                data["physical_specimen_id"] = np.asarray(metadata.get("physical_specimen_id", specimen))
            data["_source_path"] = str(path)
            data["_source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            data["_metadata"] = metadata
            validate_dataset(data)
            datasets.append(data)
    if specimens is not None:
        missing = set(specimens) - {str(d["specimen_id"]) for d in datasets}
        if missing:
            raise ValueError(f"Selected specimens not found: {sorted(missing)}")
    names = [str(d["specimen_id"]) for d in datasets]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate extraction IDs; select a single canonical dataset directory")
    return sorted(datasets, key=lambda d: (physical_id(d), str(d["specimen_id"])))


def validate_admission(metadata: dict, specimen: str) -> None:
    """Explicit quarantine/rejection cannot become training data accidentally."""
    status = " ".join(str(metadata.get(key, "")) for key in
                      ("admissibility", "admission", "admission_status", "quality_status", "status")).lower()
    if any(word in status for word in ("reject", "quarantin", "failed", "not_admitted", "not admitted")):
        raise ValueError(f"Dataset {specimen} explicitly excluded by metadata: {status}")
    for key in ("admissible", "training_admitted", "approved_for_training"):
        if metadata.get(key) is False:
            raise ValueError(f"Dataset {specimen} has {key}=false")


def validate_dataset(data: dict) -> None:
    required = {"x0", "y", "mask", "edge_index", "progress", "frame", "node_ids",
                "initial_pos_px", "scale_px", "specimen_id"}
    if required - data.keys():
        raise ValueError(f"Dataset missing fields: {sorted(required - data.keys())}")
    x0, y, mask = data["x0"], data["y"], data["mask"]
    if x0.ndim != 2 or x0.shape[1] != 2 or not len(x0):
        raise ValueError("Initial positions must be nonempty [nodes,2]")
    if y.shape != (len(data["progress"]), len(x0), 2) or mask.shape != y.shape[:2]:
        raise ValueError("Targets, masks and phase must share frame/node dimensions")
    if mask.dtype != np.dtype(bool):
        raise ValueError("Observation mask must be boolean")
    if not np.isfinite(x0).all() or not np.isfinite(data["progress"]).all():
        raise ValueError("Initial geometry and phases must be finite")
    if not np.isfinite(y[mask]).all():
        raise ValueError("Observed target values must be finite")
    scale = float(data["scale_px"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Pixel normalization scale must be finite and positive")
    if not np.isclose(np.ptp(x0[:, 1]), 1, atol=1e-4):
        raise ValueError("All geometries must be normalized by initial graph height")
    edges = data["edge_index"]
    if edges.ndim != 2 or edges.shape[0] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("Graph edges must be integer [2,edges]")
    if edges.size and (edges.min() < 0 or edges.max() >= len(x0)):
        raise ValueError("Graph edge endpoint out of bounds")
    if len(data["frame"]) != len(y) or len(np.unique(data["frame"])) != len(y):
        raise ValueError("Frame identifiers must be unique and match target frames")
    if len(data["node_ids"]) != len(x0) or len(np.unique(data["node_ids"])) != len(x0):
        raise ValueError("Node identifiers must be unique and match graph nodes")
    if not mask[compression_selection(data)].any():
        raise ValueError("A specimen needs observed compression-window targets")


def specimen_weights(datasets: list[dict]) -> list[float]:
    counts = Counter(physical_id(d) for d in datasets)
    return [1.0 / (len(counts) * counts[physical_id(d)]) for d in datasets]


def fit_affine_balanced(datasets: list[dict]) -> torch.Tensor:
    """Weighted least squares: each physical specimen contributes equally."""
    features, targets = [], []
    for data, weight in zip(datasets, specimen_weights(datasets)):
        selected = compression_selection(data)
        mask = data["mask"][selected]
        x0 = torch.as_tensor(data["x0"], dtype=torch.float64)
        progress = torch.as_tensor(data["progress"][selected], dtype=torch.float64)
        x = affine_features(x0, progress).numpy()[mask]
        factor = np.sqrt(weight / len(x))
        features.append(x * factor)
        targets.append(data["y"][selected][mask].astype(np.float64) * factor)
    coefficients = np.linalg.lstsq(np.concatenate(features), np.concatenate(targets), rcond=1e-10)[0]
    return torch.as_tensor(coefficients, dtype=torch.float32)


def model_mse(model: DeformationModel, data: dict, tensors: dict) -> torch.Tensor:
    selection = compression_selection(data)
    mask = tensors["mask"][selection]
    prediction = model(tensors["x0"], tensors["edge_index"], tensors["progress"][selection])
    # Index before subtraction so missing NaN targets cannot poison gradients.
    return (prediction[mask] - tensors["y"][selection][mask]).square().mean()


def balanced_mse(model: DeformationModel, datasets: list[dict], tensors: list[dict]) -> torch.Tensor:
    return sum(weight * model_mse(model, d, t)
               for d, t, weight in zip(datasets, tensors, specimen_weights(datasets)))


def residual_scale_for(datasets: list[dict], coefficients: torch.Tensor) -> float:
    baseline = DeformationModel(ModelConfig(kind="mlp"), coefficients)
    with torch.no_grad():
        return max(float(balanced_mse(baseline, datasets, [tensor_data(d) for d in datasets]).sqrt()), 0.01)


def train_selected(train: list[dict], validation: list[dict], kind: str, seed: int,
                   epochs: int, patience: int, learning_rate: float) -> tuple[DeformationModel, list[dict], dict]:
    if {physical_id(d) for d in train} & {physical_id(d) for d in validation}:
        raise ValueError("Training and validation physical specimens overlap")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    coefficients = fit_affine_balanced(train)
    residual_scale = residual_scale_for(train, coefficients)
    # Computing the baseline scale must not consume the initialization seed.
    torch.manual_seed(seed)
    model = DeformationModel(ModelConfig(kind=kind), coefficients, residual_scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
    train_tensors, val_tensors = [tensor_data(d) for d in train], [tensor_data(d) for d in validation]
    weights = specimen_weights(train)
    history, best_value, best_epoch = [], float("inf"), 0
    best_state = copy.deepcopy(model.state_dict())
    started = time.monotonic()
    for epoch in range(epochs + 1):
        if epoch:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            # Accumulate each graph's weighted gradient, keeping memory bounded.
            for d, t, weight in zip(train, train_tensors, weights):
                (weight * model_mse(model, d, t) / residual_scale ** 2).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                train_mse = float(balanced_mse(model, train, train_tensors))
                validation_mse = float(balanced_mse(model, validation, val_tensors))
            history.append({"epoch": epoch, "train_mse_normalized": train_mse,
                            "validation_mse_normalized": validation_mse})
            if validation_mse < best_value - 1e-10:
                best_value, best_epoch = validation_mse, epoch
                best_state = copy.deepcopy(model.state_dict())
            if epoch - best_epoch >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    info = {"seed": seed, "selected_epoch": best_epoch, "completed_epochs": epoch,
            "selected_validation_mse_normalized": best_value,
            "runtime_seconds": time.monotonic() - started, "residual_scale": residual_scale,
            **model.metadata()}
    return model, history, info


def safe_metrics(prediction: np.ndarray, data: dict, selection: np.ndarray,
                 mask_override: np.ndarray | None = None) -> dict:
    mask = data["mask"][selection] if mask_override is None else mask_override
    if not mask.any():
        return {"observed_node_frames": 0, "rmse_normalized": None, "rmse_px": None}
    return metrics(prediction[selection], data["y"][selection], mask, float(data["scale_px"]))


def evaluate_expanded(data: dict, prediction: np.ndarray) -> dict:
    progress = data["progress"]
    selections = {"compression_window": compression_selection(data),
                  "early_phase": (progress >= 0) & (progress <= 0.75),
                  "late_phase": (progress >= 0.80) & (progress <= 1)}
    result = {name: safe_metrics(prediction, data, selection) for name, selection in selections.items()}
    selected = selections["compression_window"]
    complete = data["mask"][selected].all(axis=0)
    result["complete_track_subset"] = {
        "nodes": int(complete.sum()),
        **safe_metrics(prediction, data, selected, data["mask"][selected] & complete[None])}
    result["framewise"] = [
        {"frame": int(data["frame"][i]), "phase": float(progress[i]),
         **safe_metrics(prediction, data, np.arange(len(progress)) == i)}
        for i in np.flatnonzero(selected)]
    return result


def conservative_continuity_mask(data: dict, maximum_missing_frames: int = 2) -> np.ndarray:
    """Retain each source track only until its first long observation gap.

    This is an evaluation sensitivity subset, not verified physical identity.
    It does not alter source labels or revive a missing track.
    """
    keep = data["mask"].copy()
    for node in range(keep.shape[1]):
        observed = np.flatnonzero(keep[:, node])
        if not len(observed):
            continue
        if int(data["frame"][observed[0]]) > maximum_missing_frames:
            keep[:, node] = False
            continue
        # A separation of three indices contains two unobserved frames.
        gaps = np.flatnonzero(np.diff(data["frame"][observed]) - 1 > maximum_missing_frames)
        if len(gaps):
            keep[observed[int(gaps[0]) + 1]:, node] = False
    return keep


def evaluate_saved_continuity(datasets: list[dict], output: Path, maximum_missing_frames: int = 2) -> dict:
    """Re-score fixed saved predictions without fitting or checkpoint selection."""
    summary = json.loads((output / "metrics.json").read_text())
    if len(summary["folds"]) != len(summary["protocol"]["folds"]):
        raise ValueError("Continuity sensitivity requires every registered test fold to have completed")
    lookup = {str(d["specimen_id"]): d for d in datasets}
    result = {"mask_rule": f"source observed mask, permanently truncated after first gap with >{maximum_missing_frames} missing intervening frames (observed frame separation >{maximum_missing_frames + 1})",
              "interpretation": "evaluation-only continuity sensitivity; retained identities are not independently verified",
              "training_or_selection_changed": False, "specimens": [], "folds": []}
    for specimen, data in lookup.items():
        selection = compression_selection(data)
        mask = conservative_continuity_mask(data, maximum_missing_frames)
        result["specimens"].append({"specimen_id": specimen, "physical_specimen_id": physical_id(data),
                                    "source_observed_compression_node_frames": int(data["mask"][selection].sum()),
                                    "retained_compression_node_frames": int(mask[selection].sum())})
    for fold in summary["folds"]:
        sensitive_fold = {"held_out_physical_specimen": fold["held_out_physical_specimen"],
                          "baselines": {"phase_affine": {}, "zero_displacement": {}}, "models": {"gnn": [], "mlp": []}}
        # The baseline is stored as an unchanged buffer in every checkpoint.
        # Reuse that buffer; even the analytic baseline is not fitted again.
        checkpoint = output / f"test_{fold['held_out_physical_specimen']}" / f"gnn_seed{summary['protocol']['seeds'][0]}.pt"
        baseline_model, _ = load_checkpoint(str(checkpoint))
        coefficients = baseline_model.affine_coefficients
        for specimen in fold["test_extractions"]:
            data = copy.copy(lookup[specimen])
            data["mask"] = conservative_continuity_mask(data, maximum_missing_frames)
            t = tensor_data(data)
            with torch.no_grad():
                pred = (affine_features(t["x0"], t["progress"]) @ coefficients).numpy()
            sensitive_fold["baselines"]["phase_affine"][specimen] = evaluate_expanded(data, pred)
            sensitive_fold["baselines"]["zero_displacement"][specimen] = evaluate_expanded(data, np.zeros_like(data["y"]))
        for kind in ("gnn", "mlp"):
            for row in fold["models"][kind]:
                evaluations = {}
                for specimen in fold["test_extractions"]:
                    data = copy.copy(lookup[specimen])
                    data["mask"] = conservative_continuity_mask(data, maximum_missing_frames)
                    path = output / f"test_{fold['held_out_physical_specimen']}" / f"{kind}_seed{row['seed']}_{specimen}_predictions.npz"
                    with np.load(path, allow_pickle=False) as archive:
                        evaluations[specimen] = evaluate_expanded(data, archive["prediction"])
                sensitive_fold["models"][kind].append({"seed": row["seed"], "selected_epoch": row["selected_epoch"],
                                                       "tests": evaluations})
        result["folds"].append(sensitive_fold)
    result["aggregate"] = aggregate_results(result["folds"], summary["protocol"]["seeds"])
    save_json(output / "continuity_sensitivity.json", result)
    return result


def evaluate_frozen_checkpoint(checkpoint_path: str, datasets: list[dict], output: Path) -> dict:
    """Apply an existing checkpoint to new physical experiments without fitting."""
    model, metadata = load_checkpoint(checkpoint_path)
    trained = set(metadata.get("training_specimens", [])) | set(metadata.get("training_physical_specimens", []))
    if not trained:
        raise ValueError("Frozen external evaluation requires checkpoint training-specimen metadata")
    if any(physical_id(d) in trained or str(d["specimen_id"]) in trained for d in datasets):
        raise ValueError("External evaluation contains a physical specimen used to fit this checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    result = {"task": TASK, "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
              "checkpoint_training_specimens": sorted(trained), "fitting_or_selection_performed": False,
              "external_physical_specimens": len({physical_id(d) for d in datasets}), "specimens": []}
    for data in datasets:
        specimen = str(data["specimen_id"])
        t = tensor_data(data)
        with torch.no_grad():
            prediction = model(t["x0"], t["edge_index"], t["progress"]).numpy()
        conservative = copy.copy(data)
        conservative["mask"] = conservative_continuity_mask(data)
        result["specimens"].append({"specimen_id": specimen, "physical_specimen_id": physical_id(data),
                                    "npz_sha256": data.get("_source_sha256"),
                                    "test": evaluate_expanded(data, prediction),
                                    "continuity_sensitivity": evaluate_expanded(conservative, prediction)})
        write_prediction(output / f"{specimen}_predictions.csv", data, prediction, compression_selection(data))
        np.savez_compressed(output / f"{specimen}_predictions.npz", prediction=prediction,
                            frame=data["frame"], node_ids=data["node_ids"], mask=data["mask"])
    result["aggregate"] = {}
    for subset in ("test", "continuity_sensitivity"):
        result["aggregate"][subset] = {
            "macro_rmse_normalized": float(np.mean([row[subset]["compression_window"]["rmse_normalized"]
                                                   for row in result["specimens"]])),
            "macro_rmse_px": float(np.mean([row[subset]["compression_window"]["rmse_px"]
                                            for row in result["specimens"]]))}
    save_json(output / "metrics.json", result)
    return result


def build_protocol(datasets: list[dict], args: argparse.Namespace) -> dict:
    groups = sorted({physical_id(d) for d in datasets})
    if len(groups) < 4:
        raise ValueError("Expanded protocol needs >=4 independent physical specimens")
    if args.epochs > 300:
        raise ValueError("Registered expanded benchmark caps the epoch budget at 300")
    folds = []
    for index, test in enumerate(groups):
        validation = groups[(index + 1) % len(groups)]
        train = [group for group in groups if group not in (test, validation)]
        folds.append({"held_out_physical_specimen": test,
                      "validation_physical_specimens": [validation], "training_physical_specimens": train})
    return {"task": TASK, "independent_physical_specimens": len(groups),
            "dataset_extractions": len(datasets), "outer_split": "leave-one-physical-specimen-out",
            "validation_split": "next sorted physical specimen cyclically; whole specimen",
            "phase_support_train_validation_test": [0, 1],
            "fit_weighting": "equal physical specimen; equal extraction within each physical specimen",
            "affine_fit": "weighted least squares on fitting specimens only",
            "epoch_selection": "minimum whole-validation-specimen normalized coordinate MSE; epoch zero eligible",
            "primary_metric": "equal-specimen mean normalized coordinate RMSE, averaged across fixed seeds",
            "secondary_metric": "equal-specimen mean pixel coordinate RMSE; camera scale dependent",
            "seeds": args.seeds, "maximum_epochs": args.epochs, "patience_epochs": args.patience,
            "learning_rate": args.learning_rate, "weight_decay": 0.001,
            "model_config": asdict(ModelConfig()), "test_labels_used_for_model_selection": False,
            "label_warning": "Automated image-derived tracks are noisy supervision, not DIC/FEM ground truth.",
            "independence_warning": "Extractions, frames, nodes and optimizer seeds are not independent experiments.",
            "generalization_warning": "Known archived specimens extend the pilot; this is not a prospective locked external study.",
            "datasets": [{"specimen_id": str(d["specimen_id"]), "physical_specimen_id": physical_id(d),
                          "npz_sha256": d.get("_source_sha256"), "frames": len(d["frame"]),
                          "nodes": len(d["x0"]), "directed_edges": d["edge_index"].shape[1],
                          "compression_frames": int(compression_selection(d).sum()),
                          "observed_compression_node_frames": int(d["mask"][compression_selection(d)].sum())}
                         for d in datasets], "folds": folds}


def aggregate_results(folds: list[dict], seeds: list[int]) -> dict:
    result = {}
    for kind in ("zero_displacement", "phase_affine", "gnn", "mlp"):
        result[kind] = {}
        for metric in ("rmse_normalized", "rmse_px", "mean_endpoint_error_normalized", "mean_endpoint_error_px"):
            per_seed = []
            for seed in (seeds if kind in ("gnn", "mlp") else [None]):
                per_group = []
                for fold in folds:
                    if seed is None:
                        evaluations = fold["baselines"][kind]
                    else:
                        evaluations = next(row["tests"] for row in fold["models"][kind] if row["seed"] == seed)
                    per_group.append(float(np.mean([v["compression_window"][metric] for v in evaluations.values()])))
                per_seed.append(float(np.mean(per_group)))
            result[kind][f"macro_{metric}_mean"] = float(np.mean(per_seed))
            result[kind][f"macro_{metric}_sd_over_seeds"] = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0
        result[kind]["uncertainty_note"] = "Optimizer seed variability only; no population confidence interval"
    return result


def add_training_augmentation(protocol: dict, primary: list[dict], extra: list[dict],
                              reference: dict) -> dict:
    """Keep every original test/validation label fixed while adding fit data.

    Extra re-extractions of a validation/test physical specimen are excluded
    from that fold entirely. Only the original test extraction is scored.
    """
    for key in ("folds", "datasets", "seeds", "maximum_epochs", "patience_epochs",
                "learning_rate", "weight_decay", "model_config"):
        if protocol[key] != reference[key]:
            raise ValueError(f"Augmentation must retain registered reference {key}")
    names = [str(d["specimen_id"]) for d in primary + extra]
    if len(names) != len(set(names)):
        raise ValueError("Primary and augmentation extraction IDs must be distinct")
    for data in extra:
        metadata = data.get("_metadata", {})
        status = str(metadata.get("admissibility", "")).lower()
        if metadata.get("training_admitted") is not True and not any(word in status for word in ("admitted", "approved")):
            raise ValueError(f"Augmentation extraction {data['specimen_id']} needs explicit adjacent admission metadata")
    result = copy.deepcopy(protocol)
    result["evaluation_physical_specimens"] = protocol["independent_physical_specimens"]
    result["independent_physical_specimens"] = len({physical_id(d) for d in primary + extra})
    result["dataset_extractions"] = len(primary + extra)
    result["augmentation"] = {
        "reference_test_and_validation_unchanged": True,
        "source": "explicitly approved extra image-derived pseudo-labels; training only",
        "count_as_independent_test_specimens": False,
        "datasets": [{"specimen_id": str(d["specimen_id"]), "physical_specimen_id": physical_id(d),
                      "npz_sha256": d.get("_source_sha256"), "frames": len(d["frame"]),
                      "nodes": len(d["x0"]), "compression_frames": int(compression_selection(d).sum()),
                      "observed_compression_node_frames": int(d["mask"][compression_selection(d)].sum()),
                      "compression_frames_with_observations": int((d["mask"].any(axis=1) & compression_selection(d)).sum()),
                      "compression_phase_with_observations": [
                          float(d["progress"][d["mask"].any(axis=1) & compression_selection(d)].min()),
                          float(d["progress"][d["mask"].any(axis=1) & compression_selection(d)].max())],
                      "compression_observation_coverage": float(d["mask"][compression_selection(d)].mean()),
                      "source_admissibility": d.get("_metadata", {}).get("admissibility")}
                     for d in extra]}
    for fold in result["folds"]:
        withheld = {fold["held_out_physical_specimen"], *fold["validation_physical_specimens"]}
        admitted = [d for d in extra if physical_id(d) not in withheld]
        fold["training_physical_specimens"] = sorted(set(fold["training_physical_specimens"]) |
                                                    {physical_id(d) for d in admitted})
        fold["augmentation_extractions"] = [str(d["specimen_id"]) for d in admitted]
        fold["excluded_augmentation_extractions"] = [str(d["specimen_id"]) for d in extra
                                                     if physical_id(d) in withheld]
    return result


def fit_final_expanded(datasets: list[dict], summary: dict, output: Path,
                        learning_rate: float = 0.002) -> dict:
    """All-data demonstration checkpoint; never used for held-out scores."""
    epochs = int(round(float(np.median([row["selected_epoch"] for fold in summary["folds"]
                                       for row in fold["models"]["gnn"]]))))
    seed = int(summary["protocol"]["seeds"][0])
    coefficients = fit_affine_balanced(datasets)
    scale = residual_scale_for(datasets, coefficients)
    torch.manual_seed(seed)
    model = DeformationModel(ModelConfig(kind="gnn"), coefficients, scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
    tensors, weights = [tensor_data(d) for d in datasets], specimen_weights(datasets)
    history = []
    for epoch in range(epochs + 1):
        if epoch:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for d, t, weight in zip(datasets, tensors, weights):
                (weight * model_mse(model, d, t) / scale ** 2).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                history.append({"epoch": epoch, "train_mse_normalized": float(balanced_mse(model, datasets, tensors))})
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"config": asdict(model.config), "seed": seed, "selected_epoch": epochs,
                "epoch_rule": "rounded median expanded LOSO validation-selected GNN epochs",
                "training_specimens": [str(d["specimen_id"]) for d in datasets],
                "training_physical_specimens": sorted({physical_id(d) for d in datasets}),
                "independent_evaluation": False, "input_semantics": TASK,
                "normalization": "initial graph height; centered initial xy neural features",
                "warning": "All admitted specimens used to fit this demonstration model; use outer-fold metrics for evaluation.",
                **model.metadata()}
    torch.save({"state_dict": model.state_dict(), **metadata}, output / "model.pt")
    save_json(output / "metadata.json", metadata)
    save_json(output / "history.json", history)
    return metadata


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    primary = load_expanded(args.data, getattr(args, "specimens", None))
    if getattr(args, "evaluate_checkpoint", None):
        return evaluate_frozen_checkpoint(args.evaluate_checkpoint, primary, Path(args.output))
    if getattr(args, "continuity_only", False):
        return evaluate_saved_continuity(primary, Path(args.output))
    datasets = primary
    protocol = build_protocol(primary, args)
    if getattr(args, "augmentation_data", None):
        if not getattr(args, "augmentation_specimens", None) or not getattr(args, "reference_protocol", None):
            raise ValueError("Training augmentation requires explicit --augmentation-specimens and --reference-protocol")
        extra = load_expanded(args.augmentation_data, args.augmentation_specimens)
        reference = json.loads(Path(args.reference_protocol).read_text())
        protocol = add_training_augmentation(protocol, primary, extra, reference)
        datasets = primary + extra
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Output already contains a different protocol; choose a fresh output directory")
    save_json(protocol_path, protocol)
    if getattr(args, "protocol_only", False):
        return protocol
    summary = {"task": TASK, "protocol": protocol,
               "environment": {"torch": str(torch.__version__), "numpy": np.__version__,
                               "threads": args.threads, "device": "cpu"}, "folds": []}
    for split in protocol["folds"]:
        test_group = split["held_out_physical_specimen"]
        train = [d for d in datasets if physical_id(d) in split["training_physical_specimens"]]
        validation = [d for d in primary if physical_id(d) in split["validation_physical_specimens"]]
        test = [d for d in primary if physical_id(d) == test_group]
        fold_dir = output / f"test_{test_group}"
        fold_dir.mkdir(exist_ok=True)
        coefficients = fit_affine_balanced(train)
        fold = {**split, "training_extractions": [str(d["specimen_id"]) for d in train],
                "validation_extractions": [str(d["specimen_id"]) for d in validation],
                "test_extractions": [str(d["specimen_id"]) for d in test],
                "baselines": {"phase_affine": {}, "zero_displacement": {}}, "models": {}}
        for data in test:
            specimen = str(data["specimen_id"])
            t = tensor_data(data)
            with torch.no_grad():
                pred = (affine_features(t["x0"], t["progress"]) @ coefficients).numpy()
            fold["baselines"]["phase_affine"][specimen] = evaluate_expanded(data, pred)
            fold["baselines"]["zero_displacement"][specimen] = evaluate_expanded(data, np.zeros_like(data["y"]))
            write_prediction(fold_dir / f"{specimen}_phase_affine_predictions.csv", data, pred, compression_selection(data))
        for kind in ("gnn", "mlp"):
            fold["models"][kind] = []
            for seed in args.seeds:
                model, history, info = train_selected(train, validation, kind, seed, args.epochs,
                                                     args.patience, args.learning_rate)
                stem = f"{kind}_seed{seed}"
                torch.save({"state_dict": model.state_dict(), "config": asdict(model.config),
                            "seed": seed, "selected_epoch": info["selected_epoch"],
                            "input_semantics": TASK, **split}, fold_dir / f"{stem}.pt")
                save_json(fold_dir / f"{stem}_history.json", history)
                evaluations = {}
                for data in test:
                    specimen = str(data["specimen_id"])
                    t = tensor_data(data)
                    with torch.no_grad():
                        pred = model(t["x0"], t["edge_index"], t["progress"]).numpy()
                    evaluations[specimen] = evaluate_expanded(data, pred)
                    write_prediction(fold_dir / f"{stem}_{specimen}_predictions.csv", data, pred, compression_selection(data))
                    np.savez_compressed(fold_dir / f"{stem}_{specimen}_predictions.npz", prediction=pred,
                                        frame=data["frame"], node_ids=data["node_ids"], mask=data["mask"])
                fold["models"][kind].append({**info, "tests": evaluations})
                print(json.dumps({"test": test_group, "validation": split["validation_physical_specimens"],
                                  "model": kind, "seed": seed, "selected_epoch": info["selected_epoch"],
                                  "test_rmse_normalized": float(np.mean([e["compression_window"]["rmse_normalized"]
                                                                        for e in evaluations.values()]))}), flush=True)
        summary["folds"].append(fold)
        save_json(output / "metrics.json", summary)
    summary["aggregate"] = aggregate_results(summary["folds"], args.seeds)
    if not getattr(args, "skip_final", False):
        summary["final_demonstration_model"] = fit_final_expanded(datasets, summary, output / "final", args.learning_rate)
    save_json(output / "metrics.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", default=["data/expanded"])
    parser.add_argument("--specimens", nargs="+")
    parser.add_argument("--augmentation-data", nargs="+")
    parser.add_argument("--augmentation-specimens", nargs="+")
    parser.add_argument("--reference-protocol")
    parser.add_argument("--evaluate-checkpoint")
    parser.add_argument("--continuity-only", action="store_true")
    parser.add_argument("--output", default="results/expanded")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=75)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    args = parser.parse_args()
    result = run(args)
    if args.protocol_only:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
