# BBCSport co-labeling

This repository contains an independent implementation based on [*Co-Labeling for Multi-View Weakly Labeled Learning*](https://doi.org/10.1109/TPAMI.2015.2476813) (Xu et al., TPAMI 2016). It supports semi-supervised learning (SSL), multi-instance learning (MIL), and relative outlier detection (ROD), with BBCSport as a worked experiment.

## 1. Setup

Use Python **3.12 or newer**, available as `python`. From the parent project folder, run the following on macOS/Linux. Skip environment creation if `.venv` already exists.

```sh
cd bbc-sport-co-labeling                        # Enter the repository.
python -m venv .venv                           # Create the environment once.
source .venv/bin/activate                      # Activate it in this Terminal.
python -m pip install -r requirements.lock.txt  # Install the tested package versions.
```

Activate `.venv` again whenever you open a new Terminal. `requirements.txt` lists allowed package versions; `requirements.lock.txt` pins the exact versions used for the experiment.

## 2. Real training and evaluation

**BBCSport comparison:** download the data, train all four methods across ten splits, and export predictions for the demo.

```sh
python download_data.py                                  # Download and prepare BBCSport once.
python run_experiment.py --runs 10 --seed 0 --export-demo  # Train, evaluate, and record all four methods.
```

Each split uses ten supplied answers and **367 held-out test articles**. MAP and accuracy are saved in `outputs/experiment/results.json`.

**Save and use a model:** the same general implementation can train one BBCSport split, save its model, and evaluate it separately.

```sh
python export_bbcsport.py  # Prepare train/test NPZ files for the first BBCSport split.
python train.py --task ssl --input data/bbcsport-npz/train.npz --model outputs/model.npz  # Train and save.
python predict.py --model outputs/model.npz --input data/bbcsport-npz/test.npz --output outputs/predictions.npz  # Save predictions.
python evaluate.py --model outputs/model.npz --input data/bbcsport-npz/test.npz --output outputs/metrics.json  # Evaluate against test answers.
```

**Other datasets:** use two or more numeric feature matrices, with the same samples in the same row order in every view. Feature counts can differ between views. NPZ files contain `view_0`, `view_1`, etc., plus these training fields:

| Task | Required training fields |
| --- | --- |
| SSL | `y`: numeric/string class labels; `known_mask`: which labels are supplied. Other entries of `y` are ignored. |
| MIL | `bag_ids` and `y`: one bag label repeated for every instance in that bag. Exactly two bag classes. |
| ROD | `known_normal_mask`: reference normal samples. Supply the estimated outlier fraction with `--outlier-fraction`. |

The Python helper writes dense or SciPy sparse views in this format. For example, using your own aligned matrices and label arrays:

```python
from colabeling.io import save_dataset
save_dataset("train.npz", [view_a, view_b, view_c], y=labels, known_mask=mask)
save_dataset("test.npz", [test_a, test_b, test_c], y=test_labels)
```

Use the same training/prediction/evaluation commands with your filenames. Prediction needs no labels; evaluation requires true test `y`. MIL test files also need `bag_ids` for bag evaluation. Each test view must use the same features as its training view.

Try the other settings with the included **synthetic** three-view examples:

```sh
python make_example_data.py  # Create small SSL, MIL, and ROD train/test files.
python train.py --task mil --input data/examples/mil/train.npz --model outputs/mil.npz  # Learn from bag labels.
python train.py --task rod --outlier-fraction 0.25 --input data/examples/rod/train.npz --model outputs/rod.npz  # Learn from reference normals.
python evaluate.py --model outputs/mil.npz --input data/examples/mil/test.npz --output outputs/mil-metrics.json  # Test on unseen bags.
python evaluate.py --model outputs/rod.npz --input data/examples/rod/test.npz --output outputs/rod-metrics.json  # Test outlier detection.
```

Training defaults to the full three-layer model. `--layers 1` selects flat weights; `--layers 2` combines the source/bias groups. With two views, layers 2 and 3 coincide. `--kernel rbf` and `--normalize l1` are optional. `python train.py --help` lists the remaining options. Python users can import `CoLabelingClassifier` from `colabeling`.

Optional code checks, separate from evaluating the trained models:

```sh
python -m pytest -q  # Check the solver, all three learning settings, and saved results.
```

## 3. Demo

**Preparation:** a completed animation is included. To regenerate it after training, run:

```sh
python -m pip install -r requirements-presentation.txt  # Install the optional video and chart tools.
python render_presentation.py                          # Export the 90-second animation and results chart.
```

The export uses saved predictions and observes one focused calculation through the unchanged solver. It does not repeat the ten-run experiment. The MP4 and chart are saved in `demo/`.

**Run:** on macOS, double-click **`start_demo.command`**, or run these commands from the repository:

```sh
source .venv/bin/activate  # Activate the environment in this Terminal.
python run_demo.py        # Start the local demo and open the browser.
```

Open **[http://127.0.0.1:8765](http://127.0.0.1:8765)** if the browser does not open automatically. Use **Play**, **Pause**, **Restart**, or the scene buttons. The animation follows real predictions, exchanged sheets, and learned weights. **Explore the saved data** opens the detailed explorer. Keep the Terminal open; press **Ctrl+C** there to stop.

## 4. Dataset and attribution

[BBCSport](https://derekgreene.com/bbc/): 737 articles across five sports, provided by Derek Greene and Pádraig Cunningham (ICML 2006). Source articles are copyright BBC; the dataset is for non-commercial/research use.

This is an independent implementation of [Xu et al., *Co-Labeling for Multi-View Weakly Labeled Learning*, TPAMI 2016](https://doi.org/10.1109/TPAMI.2015.2476813). Our results are reported separately from the paper's.
