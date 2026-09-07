# Camera Vision Weighing for Livestock

This project estimates cattle live weight from Topview and Rightview depth point clouds. It uses a rule-based point-cloud processing pipeline to automatically extract hand-designed geometric body measurements, rather than relying on manually annotated body landmarks or contours.

![Cow 1 Topview and Rightview RGB images with corresponding point clouds](docs/images/cow-001-rgb-pointcloud-2x2.png)

## 1. Dataset

### 1.1 Data Source

This project uses the public CowDB dataset published by Ruchay et al. (2020).

The dataset contains RGB-D images, multi-view point clouds, manual body measurements and live weights for 154 Hereford cattle.

**Dataset paper:**  
Alexey Ruchay et al. “Accurate body measurement of live cattle using three depth cameras and non-rigid 3-D shape recovery.” *Computers and Electronics in Agriculture*, 179, 105821.

https://doi.org/10.1016/j.compag.2020.105821

### 1.2 Data Used in This Project

The original 154 cattle records were manually reviewed for body visibility and point-cloud completeness.

A subset of 61 cattle was selected. These cattle had sufficiently complete body structures for geometric measurement, with at least one visible ear used as a basic visibility requirement.

Each selected record contains:

- A usable Topview point cloud
- A usable Rightview point cloud
- A ground-truth live-weight measurement
- Sufficiently complete body contours for torso measurement

## 2. Method

Depth data from the Topview and Rightview cameras are first converted into 3D point clouds. These point clouds allow measurable cattle body dimensions—such as torso length, width, depth, projected area and estimated volume—to be extracted without direct physical contact.

These geometric measurements are then compared with the ground-truth live weights to identify features associated with body weight. Regression models use the selected features to estimate live weight. Therefore, reliable point-cloud processing and accurate calculation of body dimensions are central to the overall prediction workflow.

```text
Topview and Rightview depth data
→ 3D point-cloud generation
→ Ground removal and cattle segmentation
→ PCA body-axis alignment
→ Core torso localization
→ Body-dimension and geometric-feature extraction
→ Correlation analysis and feature selection
→ Live-weight prediction
→ Leave-one-out cross-validation
```

### 2.1 Point-cloud Processing

The raw point clouds contain not only the cattle but also the floor, fences and other surrounding structures. A ground plane is first estimated and removed. Height-based connected-component segmentation is then used to isolate the main cattle body from the remaining scene.

Because cattle may appear at different orientations in the camera view, principal component analysis (PCA) is applied to the segmented Topview point cloud. The dominant horizontal PCA axis is used as the longitudinal body axis, providing a consistent coordinate system for subsequent measurements.

### 2.2 Body-dimension Extraction

Body measurements are focused on the core torso because the head, neck, legs and tail are more strongly affected by posture, movement and incomplete visibility. After PCA alignment, the body-width profile along the longitudinal axis is analysed to identify the rear and front boundaries of the core torso.

#### Core Torso Localization

![Core torso localization](docs/images/core-torso-localization.png)

The points between the detected rear and front boundaries are retained as the core torso region. Geometric measurements are then extracted from the Topview and Rightview point clouds, including:

- Torso length
- Topview body width
- Rightview body depth
- Projected torso area
- Local torso widths and depths
- Estimated torso volume

The Topview point cloud provides the torso length, horizontal widths and projected area, while the corresponding Rightview point cloud provides vertical body-depth measurements. These complementary measurements are used individually as prediction features and are also combined to estimate torso volume.

#### Elliptical Torso-volume Estimation

The core torso is divided along its longitudinal axis into slices approximately 1 cm thick. For each slice, the cross-section is approximated as an ellipse using its Topview width and corresponding Rightview depth.

$$
V = \sum_i \frac{\pi}{4} W_i D_i \Delta x
$$

Here, $W_i$ is the Topview width of slice $i$, $D_i$ is its corresponding Rightview depth and $\Delta x$ is the slice thickness. The estimated total torso volume is obtained by summing the volumes of all slices.

![Elliptical torso-volume estimation](docs/images/elliptical-torso-volume.png)

The diagram illustrates how the Topview width and Rightview depth are combined to construct an elliptical cross-section for each torso slice.

### 2.3 Weight Prediction and Evaluation

The extracted dimensions and geometric features are assessed according to their relationships with measured live weight. Individual features and selected feature combinations are then used to construct regression models for live-weight estimation.

Prediction performance is evaluated using leave-one-out cross-validation, so each animal is predicted by a model trained on all remaining animals.

## 3. Results

### 3.1 Quantitative Results

Results on the selected 61 cattle:

| Model | Pearson r | Predictive R² | MAE |
|---|---:|---:|---:|
| Elliptical torso volume | 0.643 | 0.413 | 39.7 kg |
| Elliptical volume + Rightview median depth | **0.651** | **0.419** | **39.1 kg** |
| Elliptical volume + rump height + rump width | 0.612 | 0.370 | 41.4 kg |

The current best result is a leave-one-out cross-validation MAE of **39.1 kg**.

### 3.2 Prediction Visualization

#### Actual and Predicted Weight

![Actual versus predicted weight](docs/images/actual-vs-predicted-weight.png)

The plot compares measured cattle weight with leave-one-out cross-validation predictions.

## 4. Limitations

- Data quality in the original dataset is inconsistent: many cattle have missing head-and-neck depth data or incomplete body point clouds. Therefore, 61 cattle with relatively complete body structures were manually selected for evaluation, which may introduce sample-selection bias.
- Topview widths and Rightview depths are matched according to their relative positions along the torso, as the two camera views are not spatially calibrated. Consequently, paired measurements may not represent exactly the same physical body cross-section.
- Cattle posture and incomplete body contours can cause errors in the rule-based localization of the core torso and the resulting geometric measurements. Future work could use computer-vision methods to segment the core torso automatically while excluding the head, neck, legs and tail, potentially improving measurement robustness.
