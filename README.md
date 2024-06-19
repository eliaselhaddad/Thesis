# Comparing Compression Techniques for Deep Learning Models

## Table of Contents
- [Comparing Compression Techniques for Deep Learning Models](#comparing-compression-techniques-for-deep-learning-models)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Summary of Results](#summary-of-results)
  - [Visualization](#visualization)
    - [Model Sizes and Accuracies](#model-sizes-and-accuracies)
    - [Individual Metrics](#individual-metrics)
  - [Detailed Analysis](#detailed-analysis)
  - [Practical Applications](#practical-applications)
  - [Conclusion](#conclusion)
  - [How to Run the Project](#how-to-run-the-project)
  - [Technical Report](#technical-report)
  - [References](#references)
  - [Contributing](#contributing)

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
![Model Sizes and Accuracies](https://github.com/eliaselhaddad/Thesis/assets/86868035/c3d78cea-35db-45c8-b456-ec904c901229)

### Individual Metrics
![Size, Training Accuracy, Validation Accuracy](https://github.com/eliaselhaddad/Thesis/assets/86868035/fd6034a1-cf85-48c8-9f0f-a1bda3e7b1e6)

## Detailed Analysis
- **Model Size Reduction:** Both pruning and clustering reduced the model size by approximately 66%.
- **Maintaining Accuracy:** Despite the reduction in size, the models retained high accuracy levels, with only a minor drop in validation accuracy.

## Practical Applications
- **Mobile Devices and Wearables:** Real-time analytics, health monitoring, predictive text input.
- **IoT Devices:** Localized and immediate responses without relying on constant cloud communication.

## Conclusion
This project demonstrated that pruning and clustering are effective techniques for model compression, achieving significant reductions in model size with minimal impact on accuracy.

## How to Run the Project
1. Clone the repository:
    ```sh
    git clone https://github.com/eliaselhaddad/Thesis.git
    cd Thesis
    ```
2. Create and activate a virtual environment (optional but recommended):
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```
4. Follow the provided Jupyter notebooks to reproduce the results:
    ```sh
    jupyter notebook
    ```

## Technical Report
For more detailed information, please refer to the [technical report](./Examensarbete%20-%20Elias%20El%20Haddad%20-%20AI22%20(3).pdf).

## References
- [TensorFlow Model Optimization](https://www.tensorflow.org/model_optimization)
- [Kaggle: Pruning and Quantization in Keras](https://www.kaggle.com/code/sumn2u/pruning-and-quantization-in-keras)

## Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.
