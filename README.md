# Fire Detection Model Using CNN
## Overview
This project implements a fire detection model using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The dataset contains images categorized into two classes: fire (`1`) and no fire (`0`). The model is trained to classify these images accurately.

## Project Structure
1. **Dataset Preprocessing**
   - Images are loaded, resized to `(224, 224)`, and converted to arrays.
   - Labels are encoded into one-hot format for binary classification.

2. **Data Splitting and Normalization**
   - The dataset is split into training and testing sets (80% train, 20% test).
   - Pixel values are normalized to range `[0, 1]`.

3. **CNN Model**
   - **Architecture**:
     - Two convolutional layers with ReLU activation and max-pooling.
     - Dropout for regularization.
     - Fully connected layers with `softmax` for output.
   - **Compilation**:
     - Optimizer: Adam.
     - Loss: Categorical Crossentropy.
     - Metric: Accuracy.

4. **Training**
   - Batch size: 16.
   - Epochs: 10.
   - Validation on the test set.

5. **Evaluation**
   - Test set accuracy: **88.8%**.
   - Confusion matrix and metrics (precision, recall, F1-score) are computed.

6. **Visualization**
   - Display sample test images with predicted and actual labels.
   - Plot the confusion matrix.

## Results
- **Accuracy**: 88.8%
- **Precision, Recall, F1-Score**: 0.8113, 0.7617, 0.7828.
