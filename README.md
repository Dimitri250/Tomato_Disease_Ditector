# Tomato Leaf Disease Classification

## Project Overview
This project aims to classify four different tomato leaf diseases using machine learning models. The diseases included in the classification task are:
- Early Blight
- Septoria Leaf Spot
- Target Spot
- Mosaic Virus

The objective of the project is to explore and compare the performance of a basic neural network model with an optimized neural network model, utilizing regularization and other optimization techniques.

## Dataset
The dataset used in this project contains images of tomato leaves affected by the four diseases. The data is organized into training, validation, and test sets:
- **Training set:** Used to train the models.
- **Validation set:** Used to tune hyperparameters and assess the model's performance during training.
- **Test set:** Used to evaluate the final models on unseen data.

## Models Implemented
The following models were implemented in this project:

1. **Simple Neural Network Model:**
   - A basic neural network without any optimizations or regularization techniques.
   - Used as a baseline model to compare with the optimized model.

2. **Optimized Neural Network Model:**
   - A neural network model that applies at least three optimization techniques to improve performance:
     - **Optimization Techniques:**
       1. **Optimizer Tuning:** Switching between different optimizers like Adam and RMSprop to find the best convergence.
       2. **Regularization:** Added dropout layers to prevent overfitting.
       3. **Learning Rate Scheduler:** Implemented a learning rate reduction on the plateau to stabilize training.
   - This model aims to achieve better accuracy and faster convergence compared to the simple model.

## Implementation Details
The models were implemented using Python, and the following libraries were utilized:
- **TensorFlow/Keras:** For building and training the neural network models.
- **NumPy and Pandas:** For data manipulation and analysis.
- **Matplotlib/Seaborn:** For visualizing model performance and results.

## Results
- **Simple Model:** Achieved a baseline accuracy with slower convergence.
- **Optimized Model:** Improved accuracy and faster convergence due to the application of regularization and learning rate adjustments.

Detailed analysis and visualizations of the models' performance can be found in the `notebook.ipynb` file.

## Repository Structure


## Instructions
1. Clone this repository to your local machine.
2. Open the `notebook.ipynb` file to see the code implementation and run the cells to train the models.
3. The trained models are saved in the `saved_models/` directory and can be loaded using the code provided in the notebook.

## Key Findings
- The optimized model showed significant improvement in classification accuracy over the simple neural network.
- Regularization and learning rate adjustments played a crucial role in preventing overfitting and stabilizing training.

## How to Run the Notebook
1. Ensure you have Python and Jupyter Notebook installed on your system.
2. Install the required libraries by running:
   ```bash
   pip install -r requirements.txt
