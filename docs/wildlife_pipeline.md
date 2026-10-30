# Detect and classify wildlife photographs

Birder 0.9.0 connects animal detection to per-crop species classification. It can
use a native Birder detector or an optional Ultralytics detector, classify several
animals in one photograph, and retain ambiguous predictions for human review.
See the [research and evaluation plan](research/birder-0.9.0.md) for the rationale.

## Installation

From this source folder, with Python 3.11 or newer:

```sh
python -m pip install -e .
```

For optional YOLO support:

```sh
python -m pip install -e '.[yolo]'
```

Ultralytics has separate AGPL-3.0 / Enterprise licensing terms. It is not required
for native Birder inference. The first use of a registry model can download its
weights; subsequent cached inference can run offline. Models are executed
locally; this workflow does not upload photographs.

## YOLO detection and bird classification

```sh
birder-predict-wildlife \
  --yolo-detector yolov5nu.pt \
  --classifier mobilenet_v4_l_eu-common \
  --detector-labels bird \
  --output results/birds.jsonl \
  'data/自然观察'
```

Use `python -m birder.scripts.predict_wildlife` for the same command without an
installed entry point. `yolo26n.pt` can be substituted for a newer Ultralytics
detector. Original checkpoints trained with the separate `ultralytics/yolov5`
repository are not compatible with this adapter; use YOLOv5u or compatible
Ultralytics weights.

The example classifier covers its regional bird labels. It does not identify
every species worldwide. For mixed wildlife, supply a classifier trained on the
desired animal classes and select corresponding detector labels, for example
`--detector-labels bird cat dog`. Detection labels and species labels belong to
different models. A generic `bird` detection does not itself identify a species.

## Native Birder detection

```sh
python -m birder.scripts.predict_wildlife \
  --birder-detector d_fine_l_objects365-coco_hgnet_v2_b4_pp-imagenet22k \
  --classifier mobilenet_v4_l_eu-common \
  --detector-labels bird \
  --output results/native-birds.jsonl \
  data/photos
```

Model names must identify pretrained registry entries with the correct task.
Image encoders such as `vit_h14_pn_bioclip-v25` have no named species head and are
rejected as classifiers. Fine-tune them first, or use a complete zero-shot system
with its text encoder and taxonomy. The Python API also accepts locally loaded
models and their metadata/transforms.

## Small or distant subjects

Add `--tile-size 640 --tile-overlap 128` to run overlapping detection tiles as
well as a full-image pass. Boxes are mapped into original image coordinates and
merged with class-aware NMS. More tiles increase latency. Classification crops
use native image pixels, adding 10% context on each side by default. Change this
with `--crop-padding` and validate the setting on representative photographs.

`--batch-size` controls the number of classifier crops held on the accelerator.
`--max-detections` bounds classification work per image. Detection results are
returned to CPU and output is written after each image. `--device cpu` is the
default; `--device mps` or `--device cuda:0` selects supported accelerators.

## Output and review

Output is UTF-8 JSON Lines, one object per image. Filenames, folder names, and
species names keep their original spelling. Duplicate input paths are processed
once. Existing output is protected unless `--overwrite` is supplied.

Each row records the sample path, Birder version, model identifiers, operating
settings, image size, and detections. For each detection:

| Field | Meaning |
| --- | --- |
| `box` | Clipped detector box in original pixel coordinates: x1, y1, x2, y2. |
| `crop_box` | Integer pixel rectangle actually sent to the classifier, including context. |
| `detector_label`, `detection_score` | General object category and detector score. |
| `candidate`, `species_score`, `top_k` | Leading species prediction, score, and alternatives. |
| `species` | Accepted species name, or null when review is needed. |
| `margin` | Difference between the highest and second-highest species scores. |
| `needs_review`, `review_reasons` | Low-score / small-margin review decision. |

Image status is `ok`, `review`, or `no_detections`. No detections means the
configured detector and filters found none; it is not proof of an empty scene.
`ok` means the configured thresholds passed, not that an expert verified the
species. Scores are not multiplied into a purported joint probability.

The default minimum species score is 0.7 and margin is 0.15. These are
provisional thresholds. Unknown species can be confidently misclassified.
Select thresholds with labeled validation/calibration data before using output
for scientific counts or conservation decisions.

Corrupt images stop the run by default. `--skip-errors` instead writes explicit
`error` rows and continues. Completed rows remain readable if a run is interrupted;
the file does not imply that every input was processed. Rerun to a new file or
explicitly overwrite it. Images use their stored pixel orientation, consistently
with existing directory inference; coordinates do not imply EXIF rotation.

## Python API

```python
import birder
from birder.inference.wildlife import BirderDetector, SpeciesPolicy, WildlifePipeline

det_net, det_info, det_transform = birder.load_pretrained_model_and_transform(
    "d_fine_l_objects365-coco_hgnet_v2_b4_pp-imagenet22k"
)
classifier, cls_info, cls_transform = birder.load_pretrained_model_and_transform(
    "mobilenet_v4_l_eu-common"
)
pipeline = WildlifePipeline(
    BirderDetector(det_net, det_transform, det_info.class_to_idx),
    classifier,
    cls_transform,
    cls_info.class_to_idx,
    detector_labels=["bird"],
    policy=SpeciesPolicy(min_probability=0.7, min_margin=0.15),
)
result = pipeline("data/自然观察/鸟.png")
```

Wrap the detector in `TiledDetector(detector, tile_size=(640, 640), overlap=(128,
128))` to enable tiles. For YOLO, use
`UltralyticsDetector.from_weights("yolov5nu.pt")` instead of `BirderDetector`.
All classifier inputs must use a transform producing fixed-size image tensors;
NaFlex patch/collator inputs are not supported by this crop pipeline.

## Temperature calibration

Collect classifier logits and true class indices on a held-out calibration set
using the same detected crops, padding, and transforms as deployment. Then:

```python
import torch
from birder.inference.calibration import fit_temperature
from birder.inference.wildlife import SpeciesPolicy

# NPZ or tensor preparation is application-specific; these files hold tensors.
logits = torch.load("calibration_logits.pt", weights_only=True)
labels = torch.load("calibration_labels.pt", weights_only=True)
temperature = fit_temperature(logits, labels)
policy = SpeciesPolicy(temperature=temperature)
```

Use the fitted value as `--temperature VALUE` in the CLI. Calibration does not
change the ranking of species; it adjusts score sharpness. Do not fit it on the
final test set or reuse it after changing the model/cropping configuration
without revalidation. Assess accepted-prediction error together with the fraction
sent for review.

## Validation

```sh
python -m pytest tests/test_wildlife.py tests/test_datasets.py tests/test_dataloaders.py tests/test_inference.py
```

The optional Ultralytics test creates an architecture from packaged YAML with
random weights, without downloading pretrained weights. These tests check IO,
coordinates, crop batching, review decisions, calibration, and adapter contracts.
They do not measure wildlife recognition accuracy.
