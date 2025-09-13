# Cifar Train Infer

A complete PyTorch training and inference toolkit for CIFAR-10/CIFAR-100 datasets with Streamlit UI.

## Features

- **Training**: Train ResNet-18 models on CIFAR-10/CIFAR-100 with configurable hyperparameters
- **Inference**: Load trained models for prediction on custom images or test sets
- **Evaluation**: Comprehensive model evaluation with confusion matrix and misclassification analysis
- **Visualization**: Interactive Streamlit interface for monitoring training progress and results

## Key Components

### Training
- Dataset: CIFAR-10/CIFAR-100
- Model: Modified ResNet-18 (3x3 conv, no maxpool)
- Optimizer: SGD with Nesterov momentum
- Scheduler: Cosine annealing learning rate
- Augmentation: RandomCrop, RandomHorizontalFlip, optional RandAugment
- Mixed Precision: AMP support (FP16/BF16)
- Advanced Features: Label smoothing, weight decay, channels_last memory format

### Inference & Analysis
- Custom image prediction with Top-K results
- Test set browsing (single/batch mode)
- Confusion matrix visualization (raw/normalized)
- Misclassification analysis
- CSV export for confusion matrices

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run cifar_trainer&inference_ui_cut.py
```

## Model Performance

### CIFAR-10
- 50 epochs: ~93-95% accuracy
- 200 epochs: >95% accuracy

![Figure_1](pic\Figure_1.png)

### CIFAR-100
- 50 epochs: ~70-75% accuracy
- 200 epochs: >78% accuracy

## Project Structure

- `cifar_trainer&inference_ui_cut.py`: Main Streamlit application
- `./data/`: Dataset storage directory
- `./outputs/`: Training outputs and checkpoints
- `best.pth`: Best model checkpoint
- `meta.json`: Training configuration metadata

## License

MIT License
