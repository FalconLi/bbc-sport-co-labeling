# Measured BBCSport results

Independent reproduction using seeds **0–9**, two known labels per sport, and a separate 367-article test set for each run. These are our measured scores, not the authors' published scores.

| Method | Our MAP, mean ± sample SD | Our mean accuracy | Published MAP, mean ± SD |
|---|---:|---:|---:|
| Known answers only | 83.63 ± 4.27 | 72.89% | 80.65 ± 3.38 |
| Co-training | 91.21 ± 4.87 | 86.29% | 90.31 ± 4.89 |
| Co-labeling: one layer | 93.58 ± 3.76 | 85.34% | 94.81 ± 1.41 |
| Co-labeling: two layers | 93.76 ± 3.68 | 86.76% | 95.44 ± 1.63 |

MAP is a ranking score out of 100, averaged over the five sports. Accuracy is the percentage of correct category predictions. They measure different things.

The grouped method improved mean MAP by **10.13 points** over known answers alone and **2.55 points** over this co-training baseline. Its improvement over one-layer co-labeling was **0.18 points**. The variation across splits is substantial; this experiment does not establish statistical significance of the small grouped-versus-flat difference.

The paper uses different/unreported seeds and some incompletely reported settings. This implementation fixes C=10 before evaluation and explicitly declares its baseline, stopping and integer-rounding choices in [REPRODUCTION.md](REPRODUCTION.md).

## Validation and timing

All **1,164 actual kernel fits** (550 flat, 614 grouped) reached the requested relative saddle-gap tolerance of **1e-5**. The ten runs took about **104 seconds of training/evaluation time on the development Mac**, excluding setup and browser checks; other computers can differ.

The browser replays **seed 0**, the first predeclared seed. Its numerical predictions are saved in `demo/demo-data.js`. The grid and candidate-table examples are fixed samples chosen by document order, without consulting hidden answers. Original article text stays in ignored local data.

## Reproduce

```sh
source .venv/bin/activate
python download_data.py
python run_experiment.py --runs 10 --seed 0 --export-demo
```

Exact package versions are in `requirements.lock.txt`. Archive checksums are pinned in `colabeling/data.py`. The cached recording contains the complete experiment summary, per-seed metrics, versions, scores, and optimization diagnostics. Full local reports are in the ignored `outputs/experiment/` folder.
