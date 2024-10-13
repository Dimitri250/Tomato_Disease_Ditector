# Tomato Disease Classification

## Project Overview

This project aims to classify tomato plant diseases using machine learning models. The dataset consists of images categorized into four disease classes: Early Blight, Septoria Leaf Spot, Target Spot, and Mosaic Virus. The project involves the implementation of two neural network-based models - one without optimization techniques and another with at least three optimization techniques applied.

## Dataset

The dataset used for this project is a publicly available dataset for tomato disease classification. It is divided into two folders: `train` and `test`. Each folder contains images of tomato leaves labeled with one of the four diseases.

## Project Structure

Tomato_Disease_Classification/ ├── notebook.ipynb ├── saved_models/ │ ├── model1.pkl │ ├── model2.pkl └── README.md

### Model Architecture
## Input Layer:

The model takes in images of size 300x300 with 3 color channels (RGB).
#### 1st Convolutional Layer:

Conv2D: Applies 128 filters, each of size 3x3, to detect features. Activation function is ReLU (Rectified Linear Unit).
MaxPooling2D: Reduces the dimensionality by taking the maximum value over a 5x5 pooling window, helping to reduce computation and overfitting.
#### 2nd Convolutional Layer:

Conv2D: Applies 256 filters of size 3x3. Activation function is ReLU.
MaxPooling2D: Another pooling layer with a 5x5 window to further reduce the spatial dimensions.
#### 3rd Convolutional Layer:

Conv2D: Applies 512 filters of size 3x3. Activation function is ReLU.
MaxPooling2D: Uses a 3x3 window with a stride of 2, which further reduces the size of the feature map.
#### 4th Convolutional Layer:

Conv2D: Another set of 512 filters of size 3x3. Activation function is ReLU.
Flatten Layer:

Flattens the 3D output from the convolutional layers into a 1D vector to prepare it for the fully connected layers.
Fully Connected Layer:

Dense: 512 units with ReLU activation, which allows the model to learn complex representations.
Output Layer:

Dense: 4 units with softmax activation, providing probabilities for each of the 4 classes (Early Blight, Septoria Leaf Spot, Target Spot, Mosaic Virus).

## Models Implemented

### Simple Model
The simple model is a neural network implemented without any optimization techniques. It serves as a baseline for comparing the effects of optimization techniques on model performance.
 
### First Optimized Model
The optimized model incorporates the following three optimization techniques:


1. **Regularization**: L2 regularization is applied to reduce overfitting.
2. **Learning Rate Scheduler**: A learning rate scheduler is used to adjust the learning rate dynamically during training.
3. **Data Augmentation**: Techniques such as rotation, flipping, and scaling are used to augment the training data, increasing the model's 
2. **Optimizer**: RMSprop.

### Second Optimized Model
The optimized model incorporates the following three optimization techniques:

1. **Drop out**: Used drop out techniques.
2. **Optimizer**: Adam.


## Results and Discussion

| **Model**               | **Training Accuracy (%)** | **Validation Accuracy (%)** |
|-------------------------|--------------------------|----------------------------|
| Simple Model            | 97                       | 95                         |
| First Optimized Model   | 82                       | 67                         |
| Second Optimized Model  | 95                       | 92                         |



The simpel model shows a better perfomance in both accuracy and convergence speed compared to the optimized model. The application of regularization reduces overfitting, the learning rate scheduler ensures efficient convergence, and data augmentation enhances the model's ability to generalize to unseen data.




## Error Analysis Summary:

## Confusion Matrix graphs 
<img src="https://github.com/user-attachments/assets/0882bf34-8b05-4b13-aa19-152c3dba397b" width="300"/>
<img src="https://github.com/user-attachments/assets/b5492090-bf67-428f-9302-f964af596f82" width="300"/>
<img src="https://github.com/user-attachments/assets/6f53e97c-2116-45d0-b31d-dedcd62b96fb" width="300"/>

### Simple Model:
- **Strengths:**
  - Better at identifying **Early Blight**, with fewer misclassifications for this class.
- **Weaknesses:**
  - Heavy confusion between **Early Blight** and **Target Spot**.
  - Significant misclassification of multiple diseases as **Tomato Mosaic Virus**.
  - Requires feature improvement or fine-tuning to capture virus-specific features.

### First Optimized Model:
- **Strengths:**
  - Improved classification of **Target Spot** with fewer misclassifications.
  - Better at distinguishing **Septoria Leaf Spot** compared to Model 1.
- **Weaknesses:**
  - High confusion between **Early Blight** and **Septoria Leaf Spot**.
  - Still suffers from **Tomato Mosaic Virus** misclassification, but less than Model 1.

### Second Optimized Model(Best Performing Model):
- **Strengths:**
  - Most balanced performance overall with fewer misclassifications across all categories.
  - Clear improvement in distinguishing **Tomato Mosaic Virus** from other diseases.
  - Reduced confusion between diseases compared to the previous models.
- **Weaknesses:**
  - Some minor confusion still persists between classes, though significantly reduced.
  
## Conclusion:
- **Model 3** is the best-performing model based on error analysis, showing fewer misclassifications and better generalization across all disease categories.
- It demonstrates more balanced performance and better classification for **Tomato Mosaic Virus**, making it the recommended model for deployment.
  

## Libraries Used

- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Instructions for Running the Notebook

1. Clone the repository:
   ```bash
   git clone <your-github-repo-link>
   cd Tomato_Disease_Classification


### Key Findings

- The **Simple Model** achieved high training and validation accuracy, indicating a strong initial performance, but this might suggest potential overfitting.
- The **First Optimized Model** showed a significant drop in both training and validation accuracy, which may indicate issues with model complexity or tuning of hyperparameters.
- The **Second Optimized Model** improved substantially, reaching a balance between training and validation accuracy, suggesting better generalization to unseen data.

