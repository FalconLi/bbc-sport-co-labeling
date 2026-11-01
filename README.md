# BBCSport co-labeling

This repository contains an independent implementation based on the paper [*Co-Labeling for Multi-View Weakly Labeled Learning*](https://doi.org/10.1109/TPAMI.2015.2476813) (Xu et al., TPAMI 2016), including its BBCSport experiment.

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

Download and prepare BBCSport. Training reads the files in `data/features/` automatically; original article text goes in `data/raw/`.

```sh
python download_data.py  # Download and prepare the dataset once.
```

Train all four methods, evaluate them on separate test articles, and save their predictions for the demo:

```sh
python run_experiment.py --runs 10 --seed 0 --export-demo
```

Each run uses ten supplied training answers and tests on **367 held-out articles**. Ranking scores (MAP) and accuracy appear in the Terminal and `outputs/experiment/results.json`. This command performs both training and model evaluation.

Optional code checks, separate from evaluating the trained models:

```sh
python -m pytest -q  # Check the numerical implementation and saved results.
```

## 3. Demo

**Preparation:** finish setup and the training command above. `--export-demo` saves the real results in `demo/demo-data.js` for offline playback. A completed recording is also included in the repository.

**Run:** on macOS, double-click **`start_demo.command`**, or run these commands from the repository:

```sh
source .venv/bin/activate  # Activate the environment in this Terminal.
python run_demo.py        # Start the local demo and open the browser.
```

Open **[http://127.0.0.1:8765](http://127.0.0.1:8765)** if the browser does not open automatically. The demo replays saved training rounds, lets you inspect articles and reveal answers, and compares results. No training or internet is needed to use the demo. Keep the Terminal open; press **Ctrl+C** there to stop.

## 4. Dataset and attribution

[BBCSport](https://derekgreene.com/bbc/): 737 articles across five sports, provided by Derek Greene and Pádraig Cunningham (ICML 2006). Source articles are copyright BBC; the dataset is for non-commercial/research use.

This is an independent implementation of [Xu et al., *Co-Labeling for Multi-View Weakly Labeled Learning*, TPAMI 2016](https://doi.org/10.1109/TPAMI.2015.2476813). Our results are reported separately from the paper's.
