# Camera Vision Weighing for Livestock

Research code and derived measurements for estimating cattle body weight from multi-view depth-camera point clouds.

## Current analysis

The project extracts geometric measurements from top-view and right-view cattle point clouds, including:

- PCA-aligned body and torso regions
- projected torso area and projected volume
- longitudinal width and height profiles
- ellipse-slice torso volume using top-view width and right-view depth
- virtual chest width and local rump measurements
- linear-regression evaluation with leave-one-out cross-validation

The main analysis scripts are in `tools/`. Compact derived CSV and JSON results are retained in `cattle_3d_extraction_66/` and `outputs/`.

## Data

The raw CowDB data, point clouds, images, spreadsheets, and generated previews are intentionally excluded from this repository because of their size. See `dataset/README.md` for the dataset citation and measurement definitions.

## Reproducing the latest comparison

Run:

```bash
python3 tools/evaluate_elliptical_volume_height_virtual_chest_width_61.py
```

This compares ellipse-slice volume with additional height, chest-width, and rump measurements on the 61-cattle analysis subset.
