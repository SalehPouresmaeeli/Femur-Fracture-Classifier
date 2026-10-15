# Input data

Put the input data here. Everything in this folder except this README is git-ignored, so patient data is
never committed.

Expected layout (the file and folder names are set in `configs/data.yaml`, created from
[`configs/data.example.yaml`](../configs/data.example.yaml)):

```
data/
├── <table>.xlsx        clinical table, one row per patient and load case
└── <images>/           CT images
    ├── Hip001_.vti
    └── ...
```

**Table** columns used by the pipeline:
- `Patient_id`
- `Rotation`
- `Fracture type`
- any clinical features you list in `data.tabular_features`

**Image file names** must contain `Hip<patient id>` and the rotation. Both are used to
match each image to its row in the table.

To use data stored somewhere else without copying it, override the paths when running:

```bash
python -m femur_fx run configs/experiments --set data.excel_path=<path to table> --set data.image_dir=<path to images>
```
