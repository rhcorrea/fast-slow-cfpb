# Results

Aggregate metrics behind the figures in the main README and `docs/index.html`. These files contain no complaint text; the per-complaint predictions stay in `data/processed/` (not committed).

| File | What it holds |
|---|---|
| `eval.json` | Laya → Sol: picked threshold (0.55), val and test metrics in both class mixes, and the full threshold sweep. Written by `python -m src.eval`. |
| `eval_{fast}_to_{slow}.json` | The same for other pairs, from `python -m src.eval --fast ... --slow ...`: Qwen → Sol, TF-IDF → Sol, Laya → Qwen, TF-IDF → Qwen, and earlier variants (`tfidf-5k`, `tfidf-prior`, `laya-ft-pilot`). |
| `laya_curve.jsonl` | One line per low-data run: Laya on the Mac (`src/experiments/lowdata.py`) and TF-IDF on the same samples, 5 seeds per size (`src/experiments/tfidf_curve.py`). `balanced`, `real_raw` (as is) and `real_prior` (probabilities shifted to the real class shares). |
| `t4/t4-*.json` | The Colab T4 runs from `colab/laya_t4.ipynb`: accuracy, time per complaint and training minutes. **Their `*_ece` fields use Laya's entropy `confidence`, not the calibrated score.** |
| `t4/rescored_answer_prob.json` | The T4 runs rescored with the calibrated answer probability: ECE, mean confidence and AUROC on both scores. Use these calibration figures. |
| `t4/calib_t4-full-n50.json` | Temperature refit check for the 50-per-class T4 model. |

"Real" metrics reweight the 200-per-class eval sets to each class's share of all 3,851,415 narratives. Thresholds are always picked on validation; test is reported once.
