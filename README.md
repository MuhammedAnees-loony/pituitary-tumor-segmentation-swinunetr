# Pituitary Tumor Segmentation using SwinUNETR

A deep learning project for **pituitary tumor segmentation from MRI images** using the **SwinUNETR** architecture with **PyTorch** and **MONAI**.

The project uses the **BRISC2025 dataset** and combines normal MRI images with pituitary tumor images and their corresponding segmentation masks. The trained model predicts the tumor region at the pixel level and provides visualization and evaluation using multiple performance metrics.

## Project Overview

Pituitary tumors are abnormal growths that develop in the pituitary gland and can be identified from brain MRI scans. Accurate segmentation of the tumor region can help identify and analyze the affected area.

This project focuses on developing a deep learning-based image segmentation system that can:

* Process brain MRI images
* Identify images containing pituitary tumors
* Segment the tumor region
* Generate predicted tumor masks
* Compare predictions with ground-truth masks
* Evaluate segmentation performance
* Visualize predicted tumor regions
* Perform inference on individual MRI images

The main segmentation model used in this project is **SwinUNETR**.

---

## Objectives

The main objectives of this project are:

1. Prepare MRI images and corresponding segmentation masks.
2. Organize normal and pituitary MRI images into a unified dataset structure.
3. Preprocess MRI images for deep learning.
4. Train a SwinUNETR-based segmentation model.
5. Use data augmentation to improve model generalization.
6. Evaluate the model using Dice score and accuracy.
7. Generate confusion matrices and classification metrics.
8. Perform segmentation on individual MRI images.
9. Visualize the predicted tumor regions.

---

## Dataset

This project uses the **BRISC2025 dataset**.

The dataset contains MRI images for brain tumor-related tasks, including classification and segmentation data.

For this project, the dataset is reorganized into two categories:

* `normal`
* `pituitary`

The segmentation data is used for pituitary tumor images, while normal images are assigned blank masks because they do not contain tumor regions.

### Dataset Preparation

The original BRISC2025 dataset contains separate classification and segmentation directories.

The project creates a unified directory:

```text
pituitary_vs_normal/
│
├── train/
│   ├── images/
│   │   ├── normal/
│   │   └── pituitary/
│   │
│   └── masks/
│       ├── normal/
│       └── pituitary/
│
└── test/
    ├── images/
    │   ├── normal/
    │   └── pituitary/
    │
    └── masks/
        ├── normal/
        └── pituitary/
```

For normal MRI images, blank masks containing only zero values are generated.

For pituitary images, the corresponding segmentation masks from the dataset are copied and used as ground-truth masks.

### Dataset Size Used

The final dataset prepared in the notebook contains:

| Split                |    Normal | Pituitary |     Total |
| -------------------- | --------: | --------: | --------: |
| Training             |     1,067 |     1,457 |     2,524 |
| Testing / Validation |       140 |       300 |       440 |
| **Total**            | **1,207** | **1,757** | **2,964** |

---

## Model

### SwinUNETR

The project uses **SwinUNETR (Swin Transformer UNETR)** for image segmentation.

SwinUNETR combines the strengths of the Swin Transformer architecture with the encoder-decoder structure used in U-Net-style segmentation networks.

The model is configured for 2D image segmentation with:

```text
Input Channels: 3
Output Channels: 1
Feature Size: 48
Spatial Dimensions: 2
```

The input images are converted to three channels before being passed to the model.

---

## Technologies Used

### Programming Language

* Python

### Deep Learning

* PyTorch
* MONAI
* SwinUNETR

### Image Processing

* OpenCV
* Pillow
* NumPy

### Machine Learning / Evaluation

* Scikit-learn
* TorchMetrics

### Visualization

* Matplotlib

### Development Environment

* Google Colab
* CUDA / GPU when available

### Dataset Download

* KaggleHub

---

## Data Preprocessing

The MRI images go through several preprocessing steps before training.

### 1. Loading Images

Images and segmentation masks are loaded using MONAI's image loading transforms.

### 2. Channel Conversion

The model expects three input channels.

A custom transform is used to convert images with fewer or more channels into three channels.

```text
1 channel → repeated to 3 channels
3 channels → unchanged
More than 3 channels → first 3 channels used
```

### 3. Intensity Scaling

Image intensities are normalized using:

```python
ScaleIntensityd
```

### 4. Image Resizing

Images and masks are resized/padded/cropped to:

```text
224 × 224
```

### 5. Mask Processing

The segmentation masks are converted into binary masks.

```text
0 → background
1 → tumor
```

### 6. Data Augmentation

Training images use:

* Random horizontal flipping
* Random 90-degree rotation

These transformations are applied to both images and masks to maintain their alignment.

---

## Training Configuration

The model is trained using the following configuration:

| Parameter       | Value                |
| --------------- | -------------------- |
| Image Size      | 224 × 224            |
| Input Channels  | 3                    |
| Output Channels | 1                    |
| Batch Size      | 4–6                  |
| Epochs          | 20                   |
| Learning Rate   | 0.0001               |
| Optimizer       | Adam                 |
| Loss Function   | Dice Loss + BCE Loss |
| Device          | CUDA if available    |
| Feature Size    | 48                   |

The final training configuration in the notebook uses a batch size of 4, while another training configuration uses 6 depending on GPU memory availability.

---

## Loss Function

The training process combines two loss functions:

### Dice Loss

Dice Loss helps optimize the overlap between the predicted tumor region and the ground-truth segmentation mask.

### Binary Cross Entropy

Binary Cross Entropy with logits is used to improve pixel-level classification.

The combined loss is:

```text
Total Loss = 0.5 × Dice Loss + 0.5 × BCE Loss
```

This combination helps the model learn both segmentation overlap and pixel-level classification.

---

## Training Process

The training process follows these steps:

```text
MRI Images
     ↓
Data Preprocessing
     ↓
Data Augmentation
     ↓
SwinUNETR
     ↓
Predicted Segmentation Mask
     ↓
Dice + BCE Loss
     ↓
Backpropagation
     ↓
Adam Optimizer
     ↓
Model Update
```

The model is evaluated on the validation dataset after every epoch.

The model with the highest validation Dice score is saved as:

```text
best_swin_model.pth
```

---

## Evaluation Metrics

The project evaluates the trained model using several metrics.

### Dice Score

Dice score measures the overlap between the predicted tumor segmentation and the ground-truth mask.

A higher Dice score indicates better segmentation overlap.

### Pixel Accuracy

Pixel accuracy measures the percentage of correctly classified pixels.

### Precision

Precision measures how many predicted tumor pixels are actually tumor pixels.

### Recall

Recall measures how many actual tumor pixels were correctly identified.

### Confusion Matrix

A confusion matrix is generated to analyze the segmentation predictions.

### ROC Curve

The notebook also includes ROC-related evaluation during model evaluation.

---

## Results

During training, the best validation Dice score recorded in the notebook was approximately:

```text
Best Validation Dice: 0.8328
```

The corresponding validation accuracy was approximately:

```text
Validation Accuracy: 0.9897
```

The notebook also performs additional image-wise evaluation, where a prediction is classified as pituitary when the predicted segmentation contains a tumor region.

The reported image-wise evaluation accuracy is approximately:

```text
96.82%
```

> Note: Pixel-level segmentation accuracy and image-wise classification accuracy are different measurements and should not be directly compared as the same metric.

---

## Model Evaluation Visualizations

The notebook generates several visualizations, including:

### Training Loss vs Validation Dice

Shows the relationship between training loss and validation Dice score across epochs.

### Training vs Validation Loss

Shows the change in training and validation loss during model training.

### Validation Dice and Accuracy

Shows validation segmentation performance across training epochs.

### Confusion Matrix

The project generates an image-wise confusion matrix with:

```text
Pituitary
Normal
```

The confusion matrix is saved as:

```text
SWINUNETR_Image_confusion_mat.pdf
```

---

## Single Image Inference

The trained model can also be used to predict the tumor region of an individual MRI image.

The inference workflow is:

```text
Upload MRI Image
        ↓
Preprocessing
        ↓
SwinUNETR Model
        ↓
Sigmoid Activation
        ↓
Threshold = 0.5
        ↓
Predicted Tumor Mask
        ↓
Visualization
```

The notebook allows the user to upload:

1. An original MRI image
2. Its ground-truth mask

The model then generates a predicted segmentation mask and visualizes the results.

---

## Example Output

The inference section of the notebook visualizes:

* Original MRI image
* Ground-truth tumor mask
* Predicted tumor mask
* Tumor region overlay

This makes it possible to visually compare the model's prediction with the actual tumor segmentation.

---

## Project Structure

The recommended GitHub repository structure is:

```text
pituitary-tumor-segmentation-swinunetr/
│
├── README.md
├── Final_test_SWINUNETR.ipynb
├── requirements.txt
├── .gitignore
│
├── results/
│   ├── training_vs_validation_loss.png
│   ├── validation_accuracy.png
│   ├── confusion_matrix.png
│   └── sample_prediction.png
│
└── models/
    └── README.md
```

The complete BRISC2025 dataset is not included in this repository because of its size and dataset licensing/distribution considerations.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/pituitary-tumor-segmentation-swinunetr.git
```

Move into the project directory:

```bash
cd pituitary-tumor-segmentation-swinunetr
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Requirements

The main dependencies used by the project are:

```text
torch
monai
torchmetrics
numpy
pandas
matplotlib
opencv-python
scikit-learn
Pillow
tqdm
kagglehub
```

---

## Running the Project

The project was developed and tested using **Google Colab**.

The easiest way to run it is:

1. Open the notebook.
2. Enable GPU acceleration in Google Colab.
3. Install the required packages.
4. Download the BRISC2025 dataset.
5. Run the dataset preparation section.
6. Run the preprocessing section.
7. Train the SwinUNETR model.
8. Evaluate the model.
9. Run the inference section.

---

## Google Colab

The complete notebook can be opened directly in Google Colab:

**Google Colab Notebook:**
[Add your Colab link here]

---

## Future Improvements

Possible improvements to the project include:

* Training for more epochs
* Hyperparameter tuning
* Using a larger batch size when GPU memory allows
* Testing different image resolutions
* Comparing SwinUNETR with other segmentation architectures
* Adding more advanced data augmentation
* Using cross-validation
* Improving tumor boundary segmentation
* Adding an interactive web interface for inference
* Deploying the trained model as an API
* Adding automated MRI preprocessing
* Evaluating the model on an independent external dataset

---

## Applications

Potential applications of this type of segmentation system include:

* Medical image analysis
* Research in brain tumor segmentation
* Computer-aided medical imaging
* MRI segmentation research
* Deep learning experimentation in healthcare

This project is intended for **research and educational purposes** and is not a medical diagnostic system.

---

## Author

**Muhammed Anees**

Computer Science and Engineering

### Technologies

`Python` `PyTorch` `MONAI` `SwinUNETR` `Deep Learning` `Computer Vision` `Medical Imaging`

---

## Acknowledgements

* BRISC2025 dataset
* MONAI
* PyTorch
* SwinUNETR architecture
* Google Colab
* Kaggle dataset platform

---

## License

This repository contains the project code and notebook developed for educational and research purposes.

The BRISC2025 dataset is **not redistributed with this repository**. Please refer to the original dataset source and its applicable terms before using the dataset.
