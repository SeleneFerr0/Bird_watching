# Birder 0.9.0 validation record

Validated locally on October 29, 2026, using macOS ARM64, Python 3.12.14,
PyTorch 2.15.0, Torchvision 0.29.2, and Ultralytics 8.4.193. The
[dependency snapshot](birder-0.9.0-environment.txt) records the observed environment;
it is not a portable lockfile for every operating system. Install this source
checkout separately with `pip install -e .` after preparing dependencies.

## Automated checks

| Check | Result |
| --- | --- |
| CPU regression suite excluding the four architecture-sweep files | 259 passed, 55 skipped; 77 additional subtests passed. |
| Unicode path tests | Passed for non-English directories, accented names, Chinese characters, emoji, combining characters, both image-list dataset variants, and a spawned DataLoader worker. |
| YOLO adapter tests | Real randomly initialized YOLOv5u and YOLO26 architectures ran from packaged YAML, without pretrained downloads. |
| Pipeline tests | Passed for multiple subjects, crop batching, original-image coordinates, tile merging, invalid boxes, score/label filters, review decisions, and temperature fitting. |
| Image convenience APIs | Path-like inputs and grayscale/RGBA conversion passed for classification/detection. |
| Error handling | Unreadable images become explicit error rows only when requested; model configuration errors remain fatal. |
| Full-project mypy | Passed: 375 source files. |
| Black and isort | Passed for Python files; notebooks were not checked. |
| Pylint | No findings in changed Python files. |
| Flake8 | Changed Python files passed. The broader tree retains 403 existing whitespace diagnostics under this Python environment; the unmodified 0.8.4 snapshot had 404. |
| Bandit | No findings in the new image-loading, calibration, wildlife inference, and CLI modules. |
| Dependency consistency | `pip check` passed. |
| Documentation | Strict MkDocs build passed. |

CPU command:

```sh
MPLBACKEND=Agg python -m pytest tests \
  --ignore=tests/test_net.py \
  --ignore=tests/test_net_detection.py \
  --ignore=tests/test_net_mim.py \
  --ignore=tests/test_net_ssl.py -q
```

The architecture sweeps and training flow tests were not run. CUDA, opt-in slow
tests, and tests requiring network access were skipped where marked by the
existing suite. Torchvision emitted existing image-codec deprecation warnings;
the current decoder worked in these tests. GPU/MPS performance, Python 3.11,
Windows, and Linux have not been validated in this local run. A branch/merge
request CI job now covers the core wildlife and data-loader regressions on
Python 3.11/Linux when the repository is connected to GitLab CI.

The existing sized WebDataset pipeline failed to pickle under macOS spawn. The
release now preserves standard pipeline state in a module-level serializable
wrapper. Its existing multiple-worker batching regression passed after the fix.
Two compatibility cleanups also made NumPy embedding return types and detection
visualization color types explicit for full-project type checking.

## Pretrained integration smoke test

Loaded public `yolov5nu.pt` and `yolo26n.pt` weights, plus Birder's
`mobilenet_v3_small_1_0_il-common` classifier with 371 named outputs. Tested the
repository's `docs/img/example.jpeg` with and without overlapping detection
tiles, on CPU. The image is an annotated montage, not a labeled evaluation set.

| Detector | Tiles | Subjects returned | Image status |
| --- | --- | --- | --- |
| YOLOv5u nano | No | 0 | no_detections |
| YOLOv5u nano | Yes | 1 | ok |
| YOLO26 nano | No | 1 | review |
| YOLO26 nano | Yes | 3 | review |

The tiled YOLO26 run retained two accepted candidates and one uncertain
identification. These observations verify that weight loading, native-coordinate
cropping, classification, and review decisions execute together. They do not
establish accuracy, calibration quality, recall, or a model winner. The final CLI
was also exercised on a copy of the same example in a Unicode-named folder.

Local smoke outputs are `results/release-0.9.0-smoke.json` and
`results/release-0.9.0-cli-smoke.jsonl`. The API smoke record includes SHA-256
hashes of the pretrained files. Downloaded weights were cached in a temporary
directory and are not included in the release packages.

## Release scope

Version metadata is 0.9.0. Wheel and source archives are built locally with
`python -m build --no-isolation`. The source archive includes the research report,
usage guide, tests, and dependency snapshot. No package upload, remote repository
change, pretrained model training, or public release tag was performed.

Use the [research protocol](birder-0.9.0.md) to measure field accuracy before
choosing models and operating thresholds for a conservation deployment.
