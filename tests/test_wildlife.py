import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import Optional
from unittest.mock import Mock
from unittest.mock import patch

import torch
import torch.nn.functional as F
from PIL import Image
from PIL import ImageDraw
from torch import nn
from torchvision.transforms import v2

import birder
from birder.common.image import load_image
from birder.inference import classification
from birder.inference import detection
from birder.inference.calibration import fit_temperature
from birder.inference.wildlife import BirderDetector
from birder.inference.wildlife import ImageDetector
from birder.inference.wildlife import SpeciesPolicy
from birder.inference.wildlife import TiledDetector
from birder.inference.wildlife import UltralyticsDetector
from birder.inference.wildlife import WildlifePipeline
from birder.model_registry import Task
from birder.scripts import predict_wildlife


def transform_image(image: Image.Image) -> torch.Tensor:
    return v2.functional.pil_to_tensor(image).float() / 255


class ColorClassifier(nn.Module):
    input_channels = 3
    task = Task.IMAGE_CLASSIFICATION

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(8.0))
        self.batch_sizes: list[int] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(inputs.shape[0])
        return inputs.mean(dim=(-1, -2)) * self.scale


class FixedDetector:
    label_names = {0: "bird", 1: "person"}

    def __init__(
        self,
        boxes: Optional[list[list[float]]] = None,
        scores: Optional[list[float]] = None,
        labels: Optional[list[int]] = None,
    ) -> None:
        self.boxes = [[10, 10, 20, 20], [40, 10, 50, 20]] if boxes is None else boxes
        self.scores = [0.9] * len(self.boxes) if scores is None else scores
        self.labels = [0] * len(self.boxes) if labels is None else labels

    def __call__(self, _image: Image.Image) -> dict[str, torch.Tensor]:
        return {
            "boxes": torch.tensor(self.boxes, dtype=torch.float32).reshape(-1, 4),
            "scores": torch.tensor(self.scores, dtype=torch.float32),
            "labels": torch.tensor(self.labels, dtype=torch.int64),
        }


class TestImages(unittest.TestCase):
    def test_rgb_gray_rgba_and_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "自然_🦉.png"
            for mode in ("L", "RGBA", "RGB"):
                image = Image.new(mode, (4, 5))
                image.save(path)
                for sample in (image, path, str(path)):
                    self.assertEqual(load_image(sample).mode, "RGB")
                    self.assertEqual(load_image(sample, channels=1).mode, "L")
                self.assertEqual(image.mode, mode)
                image.getpixel((0, 0))  # Caller-owned images remain open.

    def test_single_image_classification_handles_paths_modes_and_gradients(self) -> None:
        net = ColorClassifier()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "鸟.png"
            Image.new("RGBA", (4, 5), color="red").save(path)
            out, _ = classification.infer_image(net, path, transform_image)
            self.assertEqual(out.argmax(), 0)

    def test_invalid_image_input(self) -> None:
        with self.assertRaises(TypeError):
            load_image(123)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            load_image(Image.new("RGB", (4, 5)), channels=2)

    def test_single_image_detection_handles_path_and_color_mode(self) -> None:
        class Detector(nn.Module):
            def forward(
                self, inputs: torch.Tensor, **_kwargs: Any
            ) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
                if inputs.shape[1] != 3:
                    raise ValueError("Expected RGB input")
                return (
                    [
                        {
                            "boxes": torch.tensor([[1.0, 1.0, 3.0, 4.0]]),
                            "scores": torch.tensor([0.9]),
                            "labels": torch.tensor([1]),
                        }
                    ],
                    {},
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "鸟.png"
            Image.new("L", (5, 6)).save(path)
            result = detection.infer_image(Detector(), path, transform_image)
            self.assertEqual(result["boxes"].tolist(), [[1, 1, 3, 4]])


class TestWildlife(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (80, 40), color=(0, 255, 0))
        draw = ImageDraw.Draw(self.image)
        draw.rectangle((10, 10, 19, 19), fill=(255, 0, 0))
        draw.rectangle((40, 10, 49, 19), fill=(0, 0, 255))
        self.classifier = ColorClassifier()

    def pipeline(self, detector: Optional[ImageDetector] = None, **kwargs: Any) -> WildlifePipeline:
        return WildlifePipeline(
            FixedDetector() if detector is None else detector,
            self.classifier,
            transform_image,
            {"red bird": 0, "green bird": 1, "blue bird": 2},
            crop_padding=kwargs.pop("crop_padding", 0),
            **kwargs,
        )

    def test_crops_resolve_two_subjects_and_preserve_coordinates(self) -> None:
        whole_image, _ = classification.infer_image(self.classifier, self.image, transform_image)
        self.assertEqual(whole_image.argmax(), 1)  # Background dominates the whole-image input.
        self.classifier.batch_sizes.clear()
        result = self.pipeline(batch_size=1)(self.image)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([o["species"] for o in result["objects"]], ["red bird", "blue bird"])
        self.assertEqual(result["objects"][1]["box"], [40, 10, 50, 20])
        self.assertEqual(result["image_size"], [80, 40])
        self.assertEqual(self.classifier.batch_sizes, [1, 1])
        json.dumps(result, allow_nan=False)

    def test_ambiguous_predictions_are_retained_for_review(self) -> None:
        result = self.pipeline()(Image.new("RGB", self.image.size, color="white"))
        self.assertEqual(result["status"], "review")
        obj = result["objects"][0]
        self.assertIsNone(obj["species"])
        self.assertIsNotNone(obj["candidate"])
        self.assertEqual(obj["review_reasons"], ["low_probability", "small_margin"])
        self.assertEqual(len(obj["top_k"]), 3)

    def test_temperature_changes_confidence_without_changing_candidate(self) -> None:
        normal = self.pipeline()(self.image)["objects"][0]
        scaled = self.pipeline(policy=SpeciesPolicy(temperature=20))(self.image)["objects"][0]
        self.assertEqual(normal["candidate"], scaled["candidate"])
        self.assertGreater(normal["species_score"], scaled["species_score"])
        self.assertTrue(scaled["needs_review"])

    def test_empty_detections_do_not_invoke_classifier(self) -> None:
        result = self.pipeline(FixedDetector(boxes=[]))(self.image)
        self.assertEqual(result["objects"], [])
        self.assertEqual(result["status"], "no_detections")
        self.assertEqual(self.classifier.batch_sizes, [])

    def test_filtering_clipping_padding_and_detection_limit(self) -> None:
        detector = FixedDetector(
            boxes=[[-4, -5, 5, 5], [40, 10, 50, 20], [10, 10, 20, 20], [0, 0, 0, 0], [0, 0, float("nan"), 5]],
            scores=[0.8, 0.9, 0.2, 0.95, 0.99],
            labels=[0, 1, 0, 0, 0],
        )
        result = self.pipeline(detector, detector_labels=["BIRD"], crop_padding=0.2)(self.image)
        self.assertEqual(len(result["objects"]), 1)
        self.assertEqual(result["objects"][0]["box"], [0, 0, 5, 5])
        self.assertEqual(result["objects"][0]["crop_box"], [0, 0, 6, 6])
        result = self.pipeline(max_detections=1)(self.image)
        self.assertEqual(len(result["objects"]), 1)

    def test_invalid_configuration_is_rejected(self) -> None:
        policy_cases: list[dict[str, Any]] = [
            {"temperature": 0},
            {"temperature": float("nan")},
            {"min_margin": -1},
            {"top_k": 0},
        ]
        for kwargs in policy_cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SpeciesPolicy(**kwargs)
        pipeline_cases: list[dict[str, Any]] = [{"detector_labels": ["cat"]}, {"batch_size": 0}, {"crop_padding": -1}]
        for kwargs in pipeline_cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.pipeline(**kwargs)
        with self.assertRaisesRegex(ValueError, "trained species classifier"):
            WildlifePipeline(FixedDetector(), self.classifier, transform_image, {})

    def test_classifier_output_dimension_must_match_species_mapping(self) -> None:
        pipeline = WildlifePipeline(FixedDetector(), self.classifier, transform_image, {"only one": 0})
        with self.assertRaisesRegex(ValueError, "Classifier output"):
            pipeline(self.image)

    def test_small_temperature_cannot_emit_nan_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite probabilities"):
            self.pipeline(policy=SpeciesPolicy(temperature=1e-300))(self.image)


class TestDetectorAdapters(unittest.TestCase):
    def test_ultralytics_coordinate_and_label_contract(self) -> None:
        boxes = SimpleNamespace(
            xyxy=torch.tensor([[1.0, 2.0, 8.0, 9.0]]), conf=torch.tensor([0.8]), cls=torch.tensor([14.0])
        )
        model = Mock(task="detect", names={14: "bird"})
        model.predict.return_value = [SimpleNamespace(boxes=boxes)]
        detector = UltralyticsDetector(model)
        image = Image.new("RGB", (10, 10))
        result = detector(image)
        self.assertEqual(result["labels"].dtype, torch.int64)
        self.assertEqual(result["labels"].tolist(), [14])
        self.assertEqual(result["boxes"].tolist(), [[1, 2, 8, 9]])
        self.assertIs(model.predict.call_args.kwargs["source"], image)
        with self.assertRaises(ValueError):
            UltralyticsDetector(SimpleNamespace(task="classify"))

    def test_birder_scales_boxes_back_to_original_image(self) -> None:
        class Net(nn.Module):
            def forward(
                self, _inputs: torch.Tensor, **_kwargs: Any
            ) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
                return (
                    [
                        {
                            "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
                            "scores": torch.tensor([0.8]),
                            "labels": torch.tensor([1]),
                        }
                    ],
                    {},
                )

        detector = BirderDetector(Net(), lambda image: torch.zeros(3, 10, 10), {"bird": 1})
        result = detector(Image.new("RGB", (20, 40)))
        self.assertEqual(result["boxes"].tolist(), [[2, 8, 6, 16]])
        self.assertEqual(result["boxes"].device.type, "cpu")

    def test_tiles_map_and_suppress_duplicate_boxes(self) -> None:
        class RedDetector:
            label_names = {0: "bird"}

            def __call__(self, image: Image.Image) -> dict[str, torch.Tensor]:
                pixels = transform_image(image)
                y, x = torch.where((pixels[0] > 0.9) & (pixels[1] < 0.1))
                boxes = (
                    [] if len(x) == 0 else [[float(x.min()), float(y.min()), float(x.max() + 1), float(y.max() + 1)]]
                )
                return FixedDetector(boxes=boxes)(image)

        image = Image.new("RGB", (100, 40))
        ImageDraw.Draw(image).rectangle((45, 10, 49, 19), fill="red")
        result = TiledDetector(RedDetector(), tile_size=(40, 60), overlap=(0, 20))(image)
        self.assertEqual(result["boxes"].tolist(), [[45, 10, 50, 20]])
        for kwargs in ({"tile_size": (0, 60)}, {"overlap": (-1, 20)}, {"merge_threshold": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TiledDetector(RedDetector(), **kwargs)

    @unittest.skipUnless(importlib.util.find_spec("ultralytics"), "Optional Ultralytics package not installed")
    def test_real_ultralytics_architecture_without_weight_download(self) -> None:
        from ultralytics import YOLO  # type: ignore[attr-defined] # pylint: disable=import-outside-toplevel

        for architecture in ("yolov5n.yaml", "yolo26n.yaml"):
            with self.subTest(architecture=architecture):
                detector = UltralyticsDetector(YOLO(architecture), image_size=64)
                result = detector(Image.new("RGB", (40, 32)))
                self.assertEqual(result["boxes"].shape[-1], 4)
                self.assertEqual(result["labels"].dtype, torch.int64)


class TestCalibration(unittest.TestCase):
    def test_temperature_reduces_overconfident_log_loss(self) -> None:
        logits = torch.tensor([[10.0, 0.0], [10.0, 0.0], [10.0, 0.0], [10.0, 0.0]])
        labels = torch.tensor([0, 0, 0, 1])
        temperature = fit_temperature(logits, labels)
        self.assertGreater(temperature, 1)
        self.assertLess(F.cross_entropy(logits / temperature, labels), F.cross_entropy(logits, labels))
        self.assertTrue(torch.equal(logits.argmax(1), (logits / temperature).argmax(1)))

    def test_invalid_calibration_data(self) -> None:
        for logits, labels in (
            (torch.empty(0, 2), torch.empty(0, dtype=torch.int64)),
            (torch.ones(2, 2), torch.tensor([0, 2])),
            (torch.ones(2, 2), torch.tensor([0.0, 1.0])),
            (torch.full((2, 2), float("nan")), torch.tensor([0, 1])),
        ):
            with self.subTest(logits=logits), self.assertRaises(ValueError):
                fit_temperature(logits, labels)


class TestWildlifeCLI(unittest.TestCase):
    def test_skip_errors_does_not_hide_model_configuration_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bird.png"
            Image.new("RGB", (80, 40)).save(path)
            args = predict_wildlife.get_args_parser().parse_args(
                [
                    "--yolo-detector",
                    "local.pt",
                    "--classifier",
                    "species-model",
                    "--skip-errors",
                    "--output",
                    str(Path(tmpdir) / "out.jsonl"),
                    str(path),
                ]
            )
            info = SimpleNamespace(class_to_idx={"red": 0, "green": 1, "blue": 2})
            with (
                patch.object(UltralyticsDetector, "from_weights", return_value=FixedDetector()),
                patch.object(
                    birder,
                    "load_pretrained_model_and_transform",
                    return_value=(ColorClassifier(), info, transform_image),
                ),
                patch.object(WildlifePipeline, "__call__", side_effect=ValueError("invalid classifier")),
            ):
                with self.assertRaisesRegex(ValueError, "invalid classifier"):
                    predict_wildlife.predict(args)

    def test_utf8_streaming_output_and_error_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "自然"
            root.mkdir()
            Image.new("RGB", (80, 40), color="red").save(root / "鸟.png")
            (root / "broken.png").write_bytes(b"not an image")
            output = root / "观察.jsonl"
            args = predict_wildlife.get_args_parser().parse_args(
                [
                    "--yolo-detector",
                    "local.pt",
                    "--classifier",
                    "species-model",
                    "--output",
                    str(output),
                    "--skip-errors",
                    str(root),
                ]
            )
            info = SimpleNamespace(class_to_idx={"红鸟": 0, "绿鸟": 1, "蓝鸟": 2})
            with (
                patch.object(UltralyticsDetector, "from_weights", return_value=FixedDetector()),
                patch.object(
                    birder,
                    "load_pretrained_model_and_transform",
                    return_value=(ColorClassifier(), info, transform_image),
                ),
            ):
                counts = predict_wildlife.predict(args)
            self.assertEqual(counts["ok"], 1)
            self.assertEqual(counts["error"], 1)
            text = output.read_text(encoding="utf-8")
            self.assertIn("红鸟", text)
            rows = [json.loads(line) for line in text.splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["models"]["classifier"], "species-model")
            with self.assertRaises(FileExistsError):
                predict_wildlife.predict(args)

    def test_empty_input_fails_before_loading_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = predict_wildlife.get_args_parser().parse_args(
                [
                    "--yolo-detector",
                    "local.pt",
                    "--classifier",
                    "species-model",
                    "--output",
                    str(Path(tmpdir) / "out.jsonl"),
                    tmpdir,
                ]
            )
            with (
                patch.object(UltralyticsDetector, "from_weights") as load,
                self.assertRaisesRegex(ValueError, "No supported images"),
            ):
                predict_wildlife.predict(args)
            load.assert_not_called()
