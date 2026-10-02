# IR-EAR: Ear Image-Based Gender Classification

This repository contains the source code used for the experimental evaluation of the IR-EAR ear image dataset for gender classification.

The purpose of this repository is to provide the implementation details required to reproduce the transfer-learning experiments reported in the associated research article. Several ImageNet-pretrained convolutional neural network (CNN) architectures are evaluated using a consistent training and evaluation framework.

---

## Evaluated Models

The following pretrained CNN architectures are evaluated in this study:

- VGG16
- DenseNet121
- EfficientNetB7
- EfficientNetB0
- Xception
- InceptionResNetV2
- ConvNeXt-Tiny
- MobileNetV3-Large

All models are evaluated using the same dataset partitioning strategy, augmentation procedure, classification head, optimization settings, and evaluation protocol.

---

## Dataset

The IR-EAR dataset contains ear images collected from human participants for gender classification research.

Each participant contributes two ear images corresponding to the right and left ears. Participant-level partitioning is used to ensure that images belonging to the same individual are not distributed across different subsets.

The dataset metadata contains, among other fields, the following columns:

- `ID`
- `ID-Right`
- `ID-Left`
- `Gender`

The dataset itself is not included in this repository.

Please refer to the associated publication for information regarding dataset availability, access conditions, and usage.

---

## Dataset Organization

After obtaining the dataset, organize the files as follows:

```text
data/
├── IR-EAR dataset.csv
└── images/
    ├── 1.png
    ├── 2.png
    ├── 3.png
    └── ...
````

The image filenames should correspond to the image identifiers specified in the `ID-Right` and `ID-Left` columns of the metadata file.

---

## Data Preprocessing

Before model training, the dataset is partitioned at the participant level into training, validation, and test subsets.

The following split is used:

| Subset     | Proportion |
| ---------- | ---------- |
| Training   | 80%        |
| Validation | 10%        |
| Test       | 10%        |

A fixed random seed of `42` is used for reproducibility.

The splitting procedure is performed at the participant level rather than at the image level. Therefore, both left- and right-ear images belonging to the same participant remain in the same subset. This prevents participant-level data leakage between training, validation, and test sets.

Gender labels are encoded as follows:

```text
0 = Men
1 = Women
```

The gender distribution is stratified during the participant-level split.

---

## Image Processing

All input images are resized to:

```text
224 × 224 × 3
```

Model-specific preprocessing is subsequently applied according to the requirements of the corresponding pretrained architecture.

The same preprocessing procedure is applied consistently to the training, validation, and test datasets.

---

## Data Augmentation

Data augmentation is applied only to the training set.

The augmentation pipeline consists of:

* Random rotation with a factor of `0.08`
* Random translation up to `10%` in height and width
* Random zoom of `±15%`
* Random brightness adjustment with a factor of `0.20`

No data augmentation is applied to the validation or test sets.

---

## Transfer Learning Architecture

All evaluated models use ImageNet-pretrained convolutional backbones.

The pretrained backbone is frozen during training.

A common classification head is added to each backbone:

```text
Pretrained CNN Backbone
        ↓
Global Average Pooling
        ↓
Dense Layer (512 units, ReLU)
        ↓
Dropout (0.5)
        ↓
Dense Layer (1 unit, Sigmoid)
```

Using the same classification head across all architectures allows their performance to be evaluated under a consistent experimental framework.

---

## Training Configuration

The models are trained using the following configuration:

| Parameter               | Setting              |
| ----------------------- | -------------------- |
| Input size              | 224 × 224 × 3        |
| Batch size              | 16                   |
| Optimizer               | Adam                 |
| Learning rate           | 1 × 10⁻⁴             |
| Loss function           | Binary Cross-Entropy |
| Maximum epochs          | 50                   |
| Dense layer             | 512 units            |
| Dropout                 | 0.5                  |
| Backbone                | Frozen               |
| Early stopping patience | 10                   |
| Reduce LR patience      | 5                    |
| Reduce LR factor        | 0.5                  |
| Minimum learning rate   | 1 × 10⁻⁷             |
| Random seed             | 42                   |

The best model checkpoint is selected based on validation loss.

Early stopping is used to prevent unnecessary training when the validation loss does not improve.

---

## Evaluation

The trained models are evaluated on the independent test set.

The following performance metrics are reported:

* Accuracy
* Macro Precision
* Macro Recall
* Macro F1-score
* AUC-ROC

In addition, the evaluation code generates:

* Classification reports
* Confusion matrices
* ROC curves
* Training and validation accuracy curves
* Training and validation loss curves

For binary classification, a probability threshold of `0.5` is used to obtain the final predicted class.

---

## Repository Structure

The repository is organized as follows:

```text
IR-EAR/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── config.py
│
├── src/
│   ├── prepare_data.py
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
```

### Description of the main files

**`configs/config.py`**

Contains the main project configuration, including dataset paths, image size, batch size, training parameters, and output directories.

**`src/prepare_data.py`**

Performs participant-level dataset splitting and organizes the images into training, validation, and test directories.

**`src/dataset.py`**

Handles image loading, resizing, data augmentation, model-specific preprocessing, batching, and TensorFlow dataset creation.

**`src/models.py`**

Contains the pretrained CNN architectures and the common classification head used for the experiments.

**`src/train.py`**

Trains a selected model and saves the best model checkpoint and training history.

**`src/evaluate.py`**

Loads a trained model and evaluates its performance on the independent test set.

---

## Installation

Create a Python virtual environment and install the required dependencies.

For example:

```bash
python -m venv venv
```

Activate the environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

---

## Preparing the Dataset

After placing the IR-EAR dataset in the appropriate directory, run:

```bash
python src/prepare_data.py
```

This script:

1. Loads the metadata CSV file.
2. Checks the required columns.
3. Verifies gender-label consistency for each participant.
4. Encodes gender labels.
5. Creates participant-level stratified splits.
6. Creates the training, validation, and test directories.
7. Copies the corresponding ear images into the appropriate subset.

The resulting structure will be:

```text
data/
└── split/
    ├── train/
    │   ├── 0/
    │   └── 1/
    │
    ├── validation/
    │   ├── 0/
    │   └── 1/
    │
    └── test/
        ├── 0/
        └── 1/
```

---

## Training

A model can be trained using the `train.py` script.

For example, to train VGG16:

```bash
python src/train.py --model VGG16
```

To train DenseNet121:

```bash
python src/train.py --model DenseNet121
```

Other available models are:

```text
VGG16
DenseNet121
EfficientNetB7
EfficientNetB0
Xception
InceptionResNetV2
ConvNeXt-Tiny
MobileNetV3-Large
```

For example:

```bash
python src/train.py --model EfficientNetB7
```

The best model checkpoint is saved in:

```text
models/
```

The corresponding training history is saved in:

```text
results/histories/
```

---

## Model Evaluation

After training a model, its performance can be evaluated using:

```bash
python src/evaluate.py --model VGG16
```

For example:

```bash
python src/evaluate.py --model EfficientNetB7
```

The evaluation script loads the best saved model and evaluates it on the independent test set.

The resulting metrics are saved in:

```text
results/metrics/
```

Generated figures are saved in:

```text
results/figures/
```

---

## Reproducibility

To facilitate reproducibility, the following settings are kept fixed throughout the experiments:

* Participant-level data splitting
* Gender-stratified splitting
* Random seed of `42`
* 80/10/10 train-validation-test partition
* Input resolution of `224 × 224`
* Batch size of `16`
* Identical classification head
* Frozen ImageNet-pretrained backbones
* Adam optimizer
* Learning rate of `1e-4`
* Maximum of 50 training epochs
* The specified augmentation pipeline
* A probability threshold of `0.5` for binary classification

The same experimental framework is used across the evaluated CNN architectures.

---

## Results

The numerical results reported in the associated publication were obtained using the experimental configuration described above.

The repository provides the code required to train and evaluate the models. Generated metrics and figures can be stored in the `results/` directory after running the corresponding scripts.

The reported results should be interpreted together with the dataset characteristics, experimental setup, and limitations described in the associated publication.

---

## Citation

If you use the IR-EAR dataset or the code provided in this repository, please cite the associated publication:

```text
[Add the final publication citation here.]
```

A DOI or publication link can be added here after publication.

---

## License

Please refer to the repository license and the dataset usage conditions before using or redistributing the code or dataset.

The dataset is subject to its own access and usage conditions and is not distributed as part of this repository.

---

## Contact

For questions, comments, or further information regarding the dataset, code, or experiments, please contact:

* **[Dr. Seyed Reza Kamel]** — [[rezaKamel@ieee.org](mailto:rezaKamel@ieee.org)]
* **[kosar naghavi]** — [[kosar.naghavi20@gmali.com](mailto:kosar.naghavi20@gmali.com)]
* **[Fatemeh Majidi Nasab]** — [[ferimajidinasab@gmail.com](mailto:ferimajidinasab@gmail.com)]

If you have any questions, please send an email to one of the addresses above.
