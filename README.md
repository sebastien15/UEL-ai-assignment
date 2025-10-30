# CIFAR-10 Image Classification: Custom CNN vs ResNet18

**Research comparing transfer learning (ResNet18) against a custom CNN on CIFAR-10.**

## Results

| Model | Test Accuracy |
|-------|--------------|
| Custom CNN | 84.72% |
| ResNet18 (Transfer Learning) | **93.55%** |

## Files

- `CIFAR10_Training_Evaluation.ipynb` - Main notebook (trains all models)
- `app.py` - Gradio deployment app
- `models/` - Trained model weights
- `results/` - Metrics and plots

## Training

Run on Vast.ai GPU:

```bash
jupyter notebook CIFAR10_Training_Evaluation.ipynb
```

## Deployment

Deploy to Hugging Face Spaces:

```bash
# Upload these files to HF Space:
# - app.py
# - requirements.txt
# - models/resnet18_deployment.pt
```

## Paper

Full research paper: `Artificial Intelligence & Machine Vision Assignment.md`

**Author:** Ndagijimana Sebastien
**Institution:** University of East London
**Course:** CN7023 - Artificial Intelligence & Machine Vision
