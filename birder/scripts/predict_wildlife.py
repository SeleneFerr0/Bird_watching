"""Streaming detect-then-classify inference for bird and wildlife photographs."""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

import birder
from birder.common import fs_ops
from birder.inference.wildlife import BirderDetector
from birder.inference.wildlife import ImageDetector
from birder.inference.wildlife import SpeciesPolicy
from birder.inference.wildlife import TiledDetector
from birder.inference.wildlife import UltralyticsDetector
from birder.inference.wildlife import WildlifePipeline
from birder.model_registry import Task

logger = logging.getLogger(__name__)


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Locate animals, classify each crop, and export UTF-8 JSON Lines for review",
    )
    detector_group = parser.add_mutually_exclusive_group(required=True)
    detector_group.add_argument("--birder-detector", help="pretrained Birder detection model name")
    detector_group.add_argument("--yolo-detector", help="Ultralytics weights path or name, e.g. yolov5nu.pt")
    parser.add_argument("--classifier", required=True, help="pretrained Birder classifier with a named species head")
    parser.add_argument(
        "--detector-labels", nargs="+", default=["bird"], help="detection labels to classify; defaults to bird"
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device: cpu, cuda:0, or mps")
    parser.add_argument("--detection-threshold", type=float, default=0.25)
    parser.add_argument("--min-probability", type=float, default=0.7)
    parser.add_argument("--min-margin", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature fitted on held-out crop logits")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--crop-padding", type=float, default=0.1, help="context added to each side as a box fraction")
    parser.add_argument("--batch-size", type=int, default=16, help="classifier crop batch size")
    parser.add_argument("--max-detections", type=int, default=100, help="maximum crops classified per image")
    parser.add_argument("--tile-size", type=int, help="enable overlapping square detection tiles, in pixels")
    parser.add_argument("--tile-overlap", type=int, default=128, help="tile overlap in pixels")
    parser.add_argument("--tile-merge-threshold", type=float, default=0.5, help="class-aware NMS IoU threshold")
    parser.add_argument("--yolo-image-size", type=int, default=640)
    parser.add_argument("--output", type=Path, required=True, help="destination .jsonl file")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    parser.add_argument(
        "--skip-errors", action="store_true", help="record unreadable images as error rows and continue"
    )
    parser.add_argument("data_path", nargs="+", help="image files or folders, searched recursively")
    return parser


def predict(args: argparse.Namespace) -> dict[str, int]:
    policy = SpeciesPolicy(args.min_probability, args.min_margin, args.temperature, args.top_k)
    if args.output.exists() and args.overwrite is False:
        raise FileExistsError(f"Output exists: {args.output}; choose another path or use --overwrite")
    if args.batch_size < 1 or args.max_detections < 1:
        raise ValueError("batch-size and max-detections must be positive")
    if args.tile_size is not None:
        if args.tile_size <= 0 or not 0 <= args.tile_overlap < args.tile_size:
            raise ValueError("tile-size must be positive and tile-overlap must be in [0, tile-size)")
    paths = sorted(set(fs_ops.collect_samples_from_paths(args.data_path)))
    if len(paths) == 0:
        raise ValueError("No supported images found; check the input paths")

    device = torch.device(args.device)
    detector: ImageDetector
    if args.yolo_detector is not None:
        detector = UltralyticsDetector.from_weights(
            args.yolo_detector,
            device=str(device),
            image_size=args.yolo_image_size,
            confidence=args.detection_threshold,
        )
        detector_backend = "ultralytics"
        detector_name = args.yolo_detector
    else:
        net, detector_info, transform = birder.load_pretrained_model_and_transform(args.birder_detector, device=device)
        if net.task != Task.OBJECT_DETECTION:
            raise ValueError("--birder-detector must identify an object-detection model")
        detector = BirderDetector(net, transform, detector_info.class_to_idx)
        detector_backend = "birder"
        detector_name = args.birder_detector

    if args.tile_size is not None:
        detector = TiledDetector(
            detector,
            tile_size=(args.tile_size, args.tile_size),
            overlap=(args.tile_overlap, args.tile_overlap),
            merge_threshold=args.tile_merge_threshold,
        )

    classifier, info, transform = birder.load_pretrained_model_and_transform(args.classifier, device=device)
    if classifier.task != Task.IMAGE_CLASSIFICATION:
        raise ValueError("--classifier must identify an image-classification model")
    pipeline = WildlifePipeline(
        detector,
        classifier,
        transform,
        info.class_to_idx,
        detector_labels=args.detector_labels,
        detection_threshold=args.detection_threshold,
        crop_padding=args.crop_padding,
        batch_size=args.batch_size,
        max_detections=args.max_detections,
        policy=policy,
    )
    metadata: dict[str, Any] = {
        "birder_version": birder.__version__,
        "models": {"detector_backend": detector_backend, "detector": detector_name, "classifier": args.classifier},
        "settings": {
            "species_policy": asdict(policy),
            "detector_labels": args.detector_labels,
            "detection_threshold": args.detection_threshold,
            "crop_padding": args.crop_padding,
            "max_detections": args.max_detections,
            "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap if args.tile_size is not None else None,
            "tile_merge_threshold": args.tile_merge_threshold if args.tile_size is not None else None,
            "yolo_image_size": args.yolo_image_size if args.yolo_detector is not None else None,
            "device": str(device),
        },
    }
    counts = {"ok": 0, "review": 0, "no_detections": 0, "error": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w" if args.overwrite else "x", encoding="utf-8") as handle:
        for path in tqdm(paths, unit="images"):
            try:
                result = pipeline(path)
            except OSError as exc:
                if args.skip_errors is False:
                    raise
                logger.warning(f"Unable to process {path}: {exc}")
                result = {"status": "error", "error": str(exc), "objects": []}
            counts[result["status"]] += 1
            handle.write(json.dumps({"sample": path, **metadata, **result}, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
    logger.info(f"Saved {args.output}: {counts}")
    return counts


def main() -> None:
    parser = get_args_parser()
    args = parser.parse_args()
    try:
        predict(args)
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
