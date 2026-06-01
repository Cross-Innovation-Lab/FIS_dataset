## Therapy-FIS Dataset

The Therapy-FIST dataset is available at the following link:
[https://drive.google.com/drive/folders/1lxGqRBVydlEUFyWV1PjAwBEGuktjJue4?usp=sharing](https://drive.google.com/drive/folders/1lxGqRBVydlEUFyWV1PjAwBEGuktjJue4?usp=sharing)

The detailed definitions of the FIS subscales can be found in the  PDF file(fis_subscales.pdf).
To load the data of Therapy-FIS, you can use the FISdataloader.py file.

## MM-FIS Experimental Code

This repository provides the experiment/ directory and the scripts/ directory, which contain the materials and execution scripts required to reproduce the experiments.

### Data splits (`experiment/splits/`)

We provide **three division strategies** for train / validation / test evaluation. Each strategy is stored as a JSON manifest (`split_spec`, `seed`, `group_by`, and per-split sample `ID` lists). All manifests use **`split_spec: "8:1:1"`** (≈80% / 10% / 10%) and **`seed: 42`** on `all_labels_Valid.csv`. Sample IDs follow `{session_prefix}_FIS_Time{1|2|3}_{counselor_persona}` (e.g. `AG0914_FIS_Time1_Jackson`).

| Strategy | Manifest file(s) | `group_by` | What is held out |
|----------|------------------|------------|------------------|
| **Random (default)** | `default_split.json` | `none` | No grouping: all clips are shuffled at the **sample** level, then split. Use for standard i.i.d. evaluation. |
| **Cross-subject** | `cross_subject_task1_split.json`, `cross_subject_task2_split.json` | `cross_subject` | Entire **recording sessions** (ID prefix before `_FIS_`, e.g. `EO0207`, `JS0513`). All time points and counselor personas from the same session stay in the same split—tests generalization to **unseen subjects / sessions**. |
| **Cross-scenario** | `cross_scenario_task1_split.json`, `cross_scenario_task2_split.json` | `cross_scenario` | Entire **counselor personas** (last segment of the ID, e.g. `Luke`, `John`, `Bethany`). All sessions and time points for that persona stay together—tests generalization to **unseen counselor scenarios** (who is being rated in the clip). |

**Task 1 vs task 2:** Task 1 uses counselor-only features; task 2 uses counselor + patient (dyadic) features. Cross-subject and cross-scenario manifests are provided separately for each task; the **same grouping rule** applies, while the underlying sample set follows `data.task` in the dataloader.

**How to use in training:** set `train.split_file` in your experiment JSON to the manifest path (paths relative to the repo root are supported), for example:

```json
"train": {
  "split_file": "experiment/splits/cross_subject_task1_split.json",
  "split_spec": "8:1:1",
  "seed": 42
}
```

If `split_file` is omitted, training falls back to generating a split at runtime (see `experiment/train.py`). For reproducible benchmarks, use one of the pre-built manifests above.