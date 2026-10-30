"""Detect animals, classify their crops, and retain uncertain species for review.

Detector scores and species scores are kept separate: their product is not a
calibrated probability. No weights or optional packages are loaded on import.
"""

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from typing import Optional
from typing import Protocol

import torch
from PIL import Image

from birder.common.image import load_image
from birder.data.transforms.detection import InferenceTransform
from birder.inference import detection
from birder.inference import sliding_window

Detection = dict[str, torch.Tensor]


class ImageDetector(Protocol):
    label_names: dict[int, str]

    def __call__(self, image: Image.Image, /) -> Detection: ...


def _device_dtype(net: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    tensor = next(net.parameters(), None)
    if tensor is None:
        tensor = next(net.buffers(), None)
    if tensor is None:
        return (torch.device("cpu"), torch.float32)

    return (tensor.device, tensor.dtype if tensor.is_floating_point() else torch.float32)


def _cpu_detection(result: Detection) -> Detection:
    return {key: result[key].detach().cpu() for key in ("boxes", "scores", "labels")}


class BirderDetector:
    """Adapt an existing Birder detector and its matching transform."""

    def __init__(
        self,
        net: torch.nn.Module,
        transform: Callable[..., torch.Tensor],
        class_to_idx: dict[str, int],
    ) -> None:
        if len(class_to_idx) == 0:
            raise ValueError("The detector must have named detection classes")
        self.net = net.eval()
        self.transform = transform
        self.label_names = {index: name for name, index in class_to_idx.items()}

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def __call__(self, image: Image.Image) -> Detection:
        device, dtype = _device_dtype(self.net)
        image = load_image(image, channels=getattr(self.net, "input_channels", 3))
        inputs = self.transform(image).unsqueeze(0).to(device=device, dtype=dtype)
        results = detection.infer_batch(self.net, inputs)
        results = InferenceTransform.postprocess(results, [list(inputs.shape[-2:])], [list(image.size[::-1])])
        return _cpu_detection(results[0])


class UltralyticsDetector:
    """Optional adapter for Ultralytics detection models, including YOLOv5u.

    Install ``birder[yolo]`` to use ``from_weights``. Original YOLOv5-repository
    checkpoints are not compatible with this API. Dependency and weight licenses
    remain those of Ultralytics; this module does not vendor their implementation.
    """

    def __init__(
        self,
        model: Any,
        device: str = "cpu",
        image_size: int = 640,
        confidence: float = 0.05,
    ) -> None:
        if getattr(model, "task", None) != "detect":
            raise ValueError("An Ultralytics object-detection model is required")
        if image_size <= 0 or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("image_size must be positive and confidence must be in [0, 1]")
        self.model = model
        self.device = device
        self.image_size = image_size
        self.confidence = confidence
        names = model.names
        self.label_names = dict(names) if isinstance(names, dict) else dict(enumerate(names))

    @classmethod
    def from_weights(cls, weights: str | os.PathLike[str], **kwargs: Any) -> "UltralyticsDetector":
        try:
            from ultralytics import YOLO  # type: ignore[attr-defined] # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError("Install the optional detector with: pip install 'birder[yolo]'") from exc

        return cls(YOLO(str(weights)), **kwargs)

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def __call__(self, image: Image.Image) -> Detection:
        # Passing a decoded PIL image also avoids backend-specific Unicode path IO.
        results = self.model.predict(
            source=image,
            device=self.device,
            imgsz=self.image_size,
            conf=self.confidence,
            verbose=False,
            save=False,
        )
        if len(results) != 1 or results[0].boxes is None:
            raise ValueError("Expected one object-detection result")
        boxes = results[0].boxes
        return _cpu_detection({"boxes": boxes.xyxy, "scores": boxes.conf, "labels": boxes.cls.to(torch.int64)})


class TiledDetector:
    """Reuse Birder's tile mapping and class-aware merging with either detector.

    A full-image pass preserves large objects; overlapping native-resolution crops
    can preserve small subjects that would disappear in a resized full image.
    Detection tiles run sequentially to bound accelerator memory.
    """

    def __init__(
        self,
        detector: ImageDetector,
        tile_size: tuple[int, int] = (640, 640),
        overlap: tuple[int, int] = (128, 128),
        merge_threshold: float = 0.5,
        include_full_image: bool = True,
    ) -> None:
        if any(size <= 0 for size in tile_size):
            raise ValueError("tile_size must contain positive dimensions")
        if any(value < 0 or value >= size for value, size in zip(overlap, tile_size)):
            raise ValueError("overlap must be nonnegative and smaller than tile_size")
        if not math.isfinite(merge_threshold) or not 0 < merge_threshold <= 1:
            raise ValueError("merge_threshold must be in (0, 1]")
        self.detector = detector
        self.label_names = detector.label_names
        self.tile_size = tile_size
        self.overlap = overlap
        self.merge_threshold = merge_threshold
        self.include_full_image = include_full_image

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def __call__(self, image: Image.Image) -> Detection:
        image_size = (image.height, image.width)
        windows = sliding_window._generate_windows(image_size, self.tile_size, self.overlap)
        if len(windows) == 1:
            return _cpu_detection(self.detector(image))

        detections = [_cpu_detection(self.detector(image))] if self.include_full_image else []
        for window in windows:
            result = _cpu_detection(self.detector(image.crop(window)))
            detections.append(sliding_window._map_window_detection(result, window, image_size))

        return sliding_window._merge_sliding_window_detections(
            detections, mode="nms", merge_threshold=self.merge_threshold
        )


@dataclass(frozen=True)
class SpeciesPolicy:
    """Selective classification settings; defaults are provisional, not calibrated.

    Temperature must be fitted on held-out data for the exact classifier/cropping
    configuration. Confidence and top-two margin thresholds flag ambiguity; they
    do not establish that a species is absent from the training distribution.
    """

    min_probability: float = 0.7
    min_margin: float = 0.15
    temperature: float = 1.0
    top_k: int = 3

    def __post_init__(self) -> None:
        for name in ("min_probability", "min_margin"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")


class WildlifePipeline:
    """Classify each selected detection, preserving original-pixel coordinates.

    Requires a classifier with a trained, named species head. BioCLIP image
    encoders alone produce embeddings, not species logits, and are rejected when
    their class mapping is empty. Input images use stored pixel orientation.
    """

    def __init__(
        self,
        detector: ImageDetector,
        classifier: torch.nn.Module,
        transform: Callable[..., torch.Tensor],
        class_to_idx: dict[str, int],
        *,
        detector_labels: Optional[list[str]] = None,
        detection_threshold: float = 0.25,
        crop_padding: float = 0.1,
        batch_size: int = 16,
        max_detections: int = 100,
        policy: Optional[SpeciesPolicy] = None,
    ) -> None:
        if len(class_to_idx) == 0 or sorted(class_to_idx.values()) != list(range(len(class_to_idx))):
            raise ValueError("A trained species classifier with contiguous named classes is required")
        if not math.isfinite(detection_threshold) or not 0 <= detection_threshold <= 1:
            raise ValueError("detection_threshold must be in [0, 1]")
        if not math.isfinite(crop_padding) or crop_padding < 0:
            raise ValueError("crop_padding must be nonnegative and finite")
        if batch_size < 1 or max_detections < 1:
            raise ValueError("batch_size and max_detections must be positive")

        self.detector = detector
        self.classifier = classifier.eval()
        self.transform = transform
        self.class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
        self.detection_threshold = detection_threshold
        self.crop_padding = crop_padding
        self.batch_size = batch_size
        self.max_detections = max_detections
        self.policy = SpeciesPolicy() if policy is None else policy
        self.allowed_labels: Optional[set[int]] = None
        if detector_labels is not None:
            requested = {name.casefold() for name in detector_labels}
            available = {name.casefold() for name in detector.label_names.values()}
            missing = requested - available
            if len(requested) == 0 or missing:
                raise ValueError(f"Unknown or empty detector label selection: {sorted(missing)}")
            self.allowed_labels = {idx for idx, name in detector.label_names.items() if name.casefold() in requested}

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def __call__(self, sample: Image.Image | str | os.PathLike[str]) -> dict[str, Any]:
        image = load_image(sample)
        detections = _cpu_detection(self.detector(image))
        boxes, scores, labels = detections["boxes"], detections["scores"], detections["labels"]
        if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.shape != boxes.shape[:1] or labels.shape != scores.shape:
            raise ValueError("Detector must return boxes [N,4], scores [N], and labels [N]")

        objects: list[dict[str, Any]] = []
        for idx in torch.argsort(scores, descending=True, stable=True).tolist():
            score = float(scores[idx])
            if not math.isfinite(score) or not 0 <= score <= 1 or score < self.detection_threshold:
                continue
            label = int(labels[idx])
            if self.allowed_labels is not None and label not in self.allowed_labels:
                continue
            box = boxes[idx].to(torch.float32)
            if not torch.isfinite(box).all():
                continue
            x1, y1, x2, y2 = box.tolist()
            x1, x2 = max(0.0, min(x1, image.width)), max(0.0, min(x2, image.width))
            y1, y2 = max(0.0, min(y1, image.height)), max(0.0, min(y2, image.height))
            if x2 <= x1 or y2 <= y1:
                continue
            pad_x, pad_y = (x2 - x1) * self.crop_padding, (y2 - y1) * self.crop_padding
            crop_box = [
                max(0, math.floor(x1 - pad_x)),
                max(0, math.floor(y1 - pad_y)),
                min(image.width, math.ceil(x2 + pad_x)),
                min(image.height, math.ceil(y2 + pad_y)),
            ]
            objects.append(
                {
                    "box": [x1, y1, x2, y2],
                    "crop_box": crop_box,
                    "detector_label": self.detector.label_names.get(label, str(label)),
                    "detection_score": score,
                }
            )
            if len(objects) == self.max_detections:
                break

        device, dtype = _device_dtype(self.classifier)
        channels = getattr(self.classifier, "input_channels", 3)
        for start in range(0, len(objects), self.batch_size):
            batch = objects[start : start + self.batch_size]
            inputs = torch.stack(
                [self.transform(load_image(image.crop(obj["crop_box"]), channels=channels)) for obj in batch]
            ).to(device=device, dtype=dtype)
            logits = self.classifier(inputs)
            if not isinstance(logits, torch.Tensor) or logits.shape != (len(batch), len(self.class_names)):
                raise ValueError("Classifier output must be [batch, number of named species] logits")
            if not torch.isfinite(logits).all():
                raise ValueError("Classifier returned non-finite logits")
            probabilities = (logits.float() / self.policy.temperature).softmax(dim=1).cpu()
            if not torch.isfinite(probabilities).all():
                raise ValueError("Temperature scaling produced non-finite probabilities")
            values, indexes = probabilities.topk(min(max(2, self.policy.top_k), len(self.class_names)), dim=1)
            for obj, probs, indices in zip(batch, values.tolist(), indexes.tolist()):
                margin = probs[0] - probs[1] if len(probs) > 1 else probs[0]
                reasons = []
                if probs[0] < self.policy.min_probability:
                    reasons.append("low_probability")
                if margin < self.policy.min_margin:
                    reasons.append("small_margin")
                obj.update(
                    species=None if reasons else self.class_names[indices[0]],
                    candidate=self.class_names[indices[0]],
                    species_score=probs[0],
                    margin=margin,
                    needs_review=bool(reasons),
                    review_reasons=reasons,
                    top_k=[
                        {"species": self.class_names[index], "score": prob}
                        for prob, index in list(zip(probs, indices))[: self.policy.top_k]
                    ],
                )

        return {
            "image_size": [image.width, image.height],
            "objects": objects,
            "status": (
                "no_detections" if len(objects) == 0 else "review" if any(o["needs_review"] for o in objects) else "ok"
            ),
        }
