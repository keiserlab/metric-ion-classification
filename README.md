# Metric Ion Classification (MIC)

**Metric Ion Classification (MIC)** is a machine-learning-based tool designed to classify water and ion sites in X-ray crystallography and cryo-electron microscopy (cryo-EM) maps, while also providing confidence estimates for each prediction.

The **MIC-ChimeraX-Bundle** integrates MIC directly into **UCSF ChimeraX** via both a command-line and graphical interface. This makes MIC accessible to researchers without computational expertise and provides a seamless workflow for ion classification and visualization.

---

## Key Features

### Symmetry-Aware Predictions
- Automatically applies crystal symmetry expansion to improve classification accuracy.
- Symmetry-expanded contacts are displayed as a separate model for clear visualization and toggling.

### Real-Time Annotation
- Displays classification results and confidence scores directly within ChimeraX.
- Annotations appear next to predicted ion or solvent sites for intuitive inspection.

### Customizable Workflow
- Choose from multiple fingerprint types for MIC predictions.
- Specify ions of interest so that predictions are constrained to relevant species in your system.

### Batch Processing
- Run MIC on multiple PDB models in a single session.

### Exportable Results
- Export prediction results as CSV files for downstream analysis or structure refinement.

---

## Metric Ion Classification

👉 [GitHub Repository and Tutorial](https://github.com/keiserlab/metric-ion-classification/tree/main)

---

## Authors

- Laura Shub (UCSF)  
- Wenjin (Selina) Liu (UCSF)
