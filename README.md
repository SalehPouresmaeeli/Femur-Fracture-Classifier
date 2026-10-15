# Femur Fracture Classifier

Machine-learning pipeline that classifies **hip fracture type** (neck vs trochanteric) from CT images of
the femur, combined with clinical and load-case features.

It compares k-nearest neighbours, random forests and convolutional neural networks, with optional image denoising,
texture features (LBP) and class balancing (SMOTE). Each experiment is a small YAML file and runs end to end with one
command, saving metrics, figures and the trained model.

The methods were developed in 2024; this repository is a refactored, documented release of that work.

## Installation

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
pip install -e .        # installs the femur_fx package from src/ in editable mode
```

Editable mode means changes to the code take effect without reinstalling.

TensorFlow is only needed for the CNN experiments. 

## Data

No data is included in this repository.

1. Put the clinical table and the `.vti` images in the [data/](data/README.md) folder. It is git-ignored except for
   its README, which describes the expected format.
2. Create your dataset settings from the template and set your file names there:
   ```bash
   copy configs\data.example.yaml configs\data.yaml      # Linux/macOS: cp configs/data.example.yaml configs/data.yaml
   ```
   `configs/data.yaml` is git-ignored, so your file names and excluded patients stay private.

## Usage

```bash
# run the example experiment
python -m femur_fx run configs/example.yaml

# change settings without editing files
python -m femur_fx run configs/example.yaml --set evaluation.n_splits=10

# list / run all of your own experiments, continuing after failures
python -m femur_fx list configs/experiments
python -m femur_fx run configs/experiments --keep-going

# noise analysis of the images
python -m femur_fx noise
```

Results are written to `outputs/<experiment>/<timestamp>/`.

## Pipeline

```
config ─► data ─► [hyper-parameter search] ─► evaluation ─► metrics, figures, model
           │
           ├─ table: clean labels, add loading direction
           └─ images: joined to table rows by patient and load case

KNN / random forest:   [image | features] ─► SMOTE* ─► denoise* ─► LBP* ─► classifier
CNN:                   images ─► denoise* ─► scaling* ─┐
                       features ───────────────────────┴─► SMOTE* ─► CNN
                                                                 * = optional
```

- **Splits are grouped by patient**: a model is never tested on a patient it was trained on.
- **SMOTE only resamples training data**, never the data a model is evaluated on. For the CNN it creates synthetic
  images and features together, so they stay paired.
- **Each image is denoised separately**, so one image never affects another.

## Experiments

An experiment is a YAML file. [configs/example.yaml](configs/example.yaml) is a commented template: copy it into
`configs/experiments/` and edit it. That folder is git-ignored, so your own experiments are never committed.

An experiment can inherit from another one and list only what changes:

```yaml
base: ../example.yaml
name: example_ten_folds
evaluation: {n_splits: 10}
```

All settings are documented in [src/femur_fx/config.py](src/femur_fx/config.py). CNN layers are defined in the
`model.architecture` section; see [src/femur_fx/models/cnn.py](src/femur_fx/models/cnn.py).

## Outputs

Each run folder contains:
- the settings used and a log
- metrics per split, with mean and standard deviation
- all test predictions
- figures: confusion matrix, ROC curve, probability histograms, F1 spread
- the trained model

## Project layout

```
├── configs/              templates (data.example.yaml, example.yaml); your data.yaml and experiments/ are git-ignored
├── data/                 input data (git-ignored)
├── src/
│   └── femur_fx/         the package
│       ├── config.py         reads the YAML configs
│       ├── cli.py            command line: run, list, noise
│       ├── data/             reads the images and the table, cleans labels, builds the dataset
│       ├── preprocessing/    denoising, LBP texture features, SMOTE
│       ├── models/           KNN / random forest pipelines, configurable CNN
│       ├── evaluation/       patient-grouped splits, metrics, figures
│       ├── experiments/      runs an experiment end to end and saves the results
│       └── analysis/         noise analysis of the images
└── tests/                tests on synthetic data
```

## Testing

```bash
python -m pytest
```

The tests use a small synthetic dataset, so no patient data is needed. The CNN tests are skipped if TensorFlow is
not installed.

## Troubleshooting

- **`Config file not found: configs/data.yaml`**: create it from the template (see [Data](#data)).
- **`Excel table not found` / `Image folder not found`**: the data is not where `configs/data.yaml` says. Copy it
  into `data/`, fix the names in `configs/data.yaml`, or set the paths with `--set data.excel_path=...` and
  `--set data.image_dir=...`.
- **TensorFlow install fails on Windows** (`No such file or directory ...`): a path inside TensorFlow is longer than
  Windows allows. Create the virtual environment in a short folder (e.g. `C:\venvs\femur`) or enable long paths.
- **`PyWavelets is not installed`**: run `pip install PyWavelets`.

## How to cite

If you use this software in your work, please cite it. Use the "Cite this repository" button on the GitHub page,
or see [CITATION.cff](CITATION.cff).

## License

No license. All rights are reserved.

## Contact

- Questions and bug reports: please [open an issue](https://github.com/SalehPouresmaeeli/femur-fracture-classifier/issues).
- Collaboration and industry pilots: connect with the developer on
  [LinkedIn](https://www.linkedin.com/in/saleh-pouresmaeeli).
