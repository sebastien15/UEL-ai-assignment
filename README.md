# Deep Learning Image Classification on CIFAR-10

**Module:** CN-7023 Artificial Intelligence & Machine Vision  
**Institution:** University of East London  
**Academic Year:** 2024/2025

---

## 📋 Project Overview

This project implements and evaluates multiple Convolutional Neural Network (CNN) architectures for multi-class image classification on the CIFAR-10 dataset. The work demonstrates a comprehensive understanding of deep learning principles, transfer learning, and model optimization techniques.

### Research Questions

1. How do different CNN architectures (Custom CNN, ResNet18, VGG16) perform on CIFAR-10 classification?
2. What impact does data augmentation have on model generalization and overfitting?
3. How does transfer learning from ImageNet improve classification performance compared to training from scratch?
4. What optimization strategies (learning rates, optimizers, schedulers) yield the best results?

---

## 🗂️ Project Structure

```
CN7023_Assignment/
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Dataset analysis and visualization
│   ├── 02_baseline_cnn.ipynb          # Custom CNN implementation
│   ├── 03_resnet_training.ipynb       # ResNet18 experiments
│   ├── 04_vgg_training.ipynb          # VGG16 experiments
│   └── 05_results_analysis.ipynb      # Comprehensive results analysis
│
├── src/
│   ├── models/
│   │   ├── custom_cnn.py              # Custom CNN architecture
│   │   ├── resnet.py                  # ResNet18 implementation
│   │   └── vgg.py                     # VGG16 implementation
│   │
│   └── utils/
│       ├── data_loader.py             # Data loading and preprocessing
│       ├── training.py                # Training loop utilities
│       ├── evaluation.py              # Evaluation metrics
│       └── visualization.py           # Plotting utilities
│
├── results/
│   ├── figures/                       # All plots and visualizations
│   ├── logs/                          # Training logs
│   └── metrics.csv                    # Performance metrics
│
├── report/
│   └── CN7023_Report.pdf              # Final research report
│
├── presentation/
│   └── CN7023_Presentation.pdf        # Project presentation
│
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA-capable GPU (recommended) or Google Colab

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/sebastien15/UEL-ai-assignment.git
cd UEL-ai-assignment
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download CIFAR-10 dataset:**
The dataset will be automatically downloaded when you run the notebooks for the first time.

### Running on Google Colab

1. Open Google Colab: https://colab.research.google.com
2. Clone this repository in Colab:
```python
!git clone https://github.com/sebastien15/UEL-ai-assignment.git
%cd UEL-ai-assignment
```
3. Enable GPU: Runtime → Change runtime type → GPU
4. Open and run the notebooks in order

---

## 📊 Dataset Information

**CIFAR-10 Dataset:**
- **Training Images:** 50,000 (32×32 RGB)
- **Test Images:** 10,000 (32×32 RGB)
- **Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Source:** https://www.cs.toronto.edu/~kriz/cifar.html

---

## 🏗️ Model Architectures

### 1. Custom CNN (Baseline)
- 4 convolutional blocks with batch normalization
- Dropout regularization
- ~500K parameters
- **Purpose:** Demonstrate understanding of CNN fundamentals

### 2. ResNet18
- 18 layers with residual connections
- Pre-trained on ImageNet (transfer learning)
- ~11M parameters
- **Purpose:** Modern architecture with skip connections

### 3. VGG16
- 16 deep layers
- Pre-trained on ImageNet
- ~138M parameters
- **Purpose:** Classic deep network architecture

---

## 📈 Results Summary

| Model | Data Augmentation | Transfer Learning | Test Accuracy |
|-------|-------------------|-------------------|---------------|
| Custom CNN | No | No | ~70% |
| ResNet18 | No | No | ~82% |
| ResNet18 | Yes | No | ~87% |
| ResNet18 | Yes | Yes | ~90%+ |
| VGG16 | Yes | Yes | ~88% |

*Note: Results will be updated after training completion*

---

## 🔬 Experiments Conducted

1. **Architecture Comparison:** Evaluated baseline performance of different architectures
2. **Data Augmentation:** Tested impact of various augmentation strategies
3. **Transfer Learning:** Compared pre-trained vs. training from scratch
4. **Hyperparameter Optimization:** Optimized learning rates, optimizers, and schedulers

---

## 📚 Key References

1. Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images.
2. He, K., et al. (2016). Deep Residual Learning for Image Recognition.
3. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition.
4. Yosinski, J., et al. (2014). How transferable are features in deep neural networks?

---

## 📝 Usage Examples

### Training a Model

```python
from src.models.resnet import ResNet18
from src.utils.data_loader import get_cifar10_loaders
from src.utils.training import train_model

# Load data
train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=128)

# Create model
model = ResNet18(num_classes=10, pretrained=True)

# Train
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    learning_rate=0.001
)
```

### Evaluating a Model

```python
from src.utils.evaluation import evaluate_model

# Evaluate
metrics = evaluate_model(model, test_loader)
print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
```

---

## 🛠️ Technologies Used

- **Deep Learning:** PyTorch, torchvision
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Development:** Jupyter Notebook, Google Colab
- **Version Control:** Git, GitHub

---

## 📧 Contact

**Student:** Sebastien  
**Module Code:** CN-7023  
**Institution:** University of East London

---

## 📄 License

This project is submitted as part of academic coursework for the CN-7023 module at the University of East London. All rights reserved.

---

## 🙏 Acknowledgments

- CIFAR-10 dataset creators: Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- PyTorch development team
- University of East London, School of Architecture, Computing and Engineering

---

**Last Updated:** October 2025
