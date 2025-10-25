# CN-7023 Assignment - CIFAR-10 Image Classification

**Student:** Sebastien  
**Module:** CN-7023 Artificial Intelligence & Machine Vision  
**University of East London**

## Project Description

This project implements different CNN architectures for image classification on the CIFAR-10 dataset as part of my coursework assignment.

## What's Included

- **notebooks/** - Jupyter notebooks with all the code
  - 01_data_exploration.ipynb - Dataset analysis
  - 02_baseline_cnn.ipynb - Custom CNN model
  - 03_resnet_training.ipynb - ResNet18 implementation
  - 04_vgg_training.ipynb - VGG16 implementation
  - 05_results_analysis.ipynb - Results comparison

- **results/** - Training results and figures
- **report/** - Final report PDF
- **presentation/** - Presentation slides

## How to Run

### Using Google Colab 
1. Open https://colab.research.google.com
2. Clone this repo: `!git clone https://github.com/sebastien15/UEL-ai-assignment.git`
3. Enable GPU (Runtime → Change runtime type → GPU)
4. Run the notebooks in order

### Local Setup
```bash
pip install -r requirements.txt
jupyter notebook
```

## Dataset

CIFAR-10: 60,000 32x32 color images in 10 classes
- Training: 50,000 images
- Test: 10,000 images
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Models Tested

1. **Custom CNN** - Baseline model (~500K parameters)
2. **ResNet18** - With transfer learning (~11M parameters)
3. **VGG16** - Deep network (~138M parameters)

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib
- jupyter

See `requirements.txt` for full list.

## References

- Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images
- He, K., et al. (2016). Deep Residual Learning for Image Recognition
- Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks

---

*This is coursework for CN-7023 module at UEL By Ndagijimana Sebastien*
