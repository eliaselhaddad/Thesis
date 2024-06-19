# Comparing Compression Techniques for Deep Learning Models

## Introduction
This project compares various model optimization techniques, specifically pruning, clustering, and quantization, to evaluate their impact on model size and performance.

## Summary of Results
| Model                | Size (KB) | Training Accuracy | Validation Accuracy | Test Accuracy (Quantized) |
|----------------------|-----------|-------------------|---------------------|--------------------------|
| **Teacher Model**    | 4075.12   | 99.55%            | 90.10%              | 91.57%                   |
| **Pruned Model**     | 1365.81   | 99.36%            | 89.23%              | 90.30%                   |
| **Clustered Model**  | 1365.78   | 98.85%            | 89.27%              | 90.03%                   |

## Visualization
Here are some visual comparisons of the model sizes and accuracies:

### Model Sizes and Accuracies
![Model Sizes and Accuracies](comparison.png)

### Individual Metrics
![Size, Training Accuracy, Validation Accuracy](size-training-validation.png)

## Detailed Analysis
- **Model Size Reduction:** Both pruning and clustering reduced the model size by approximately 66%.
- **Maintaining Accuracy:** Despite the reduction in size, the models retained high accuracy levels, with only a minor drop in validation accuracy.

## Practical Applications
- **Mobile Devices and Wearables:** Real-time analytics, health monitoring, predictive text input.
- **IoT Devices:** Localized and immediate responses without relying on constant cloud communication.

## Conclusion
This project demonstrated that pruning and clustering are effective techniques for model compression, achieving significant reductions in model size with minimal impact on accuracy.

## How to Run the Project
1. Clone the repository.
2. Install the required dependencies.
3. Follow the provided Jupyter notebooks to reproduce the results.

## References
- [TensorFlow Model Optimization](https://www.tensorflow.org/model_optimization)
- [Kaggle: Pruning and Quantization in Keras](https://www.kaggle.com/code/sumn2u/pruning-and-quantization-in-keras)
![comparison](https://github.com/eliaselhaddad/Thesis/assets/86868035/c3d78cea-35db-45c8-b456-ec904c901229)
