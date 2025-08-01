# Image Classification Model Using CNN

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: YATIN GOGIA

*INTERN ID*: CT04DH1053

*DOMAIN*: MACHINE LEARNING

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

*DESCRIPTION* :
## Dataset  
**Dogs vs Cats dataset from Kaggle**  
[Dataset link](https://www.kaggle.com/datasets/salader/dogs-vs-cats)

## Overview

This project implements an **image classification model** using **Convolutional Neural Networks (CNNs)** to classify images of dogs and cats. The iconic "Dogs vs Cats" dataset serves as the benchmark for this binary classification task.

CNNs excel in image classification by automatically learning hierarchical spatial features from raw pixel data, outperforming traditional methods. This repository contains the code to preprocess data, define the CNN architecture, train the model, and analyze results.

---

## Project Structure

- `main.py`:  
  Handles data loading, preprocessing, training, validation, and testing.

- `model.py`:  
  Defines the CNN architecture and model compilation.

---

## Step-by-Step Explanation

### 1. Dataset: Dogs vs Cats  
- Contains 25,000 labeled images (half dogs, half cats).  
- Images vary in pose, lighting, and background complexity.  
- Filenames include labels (e.g. `cat.0.jpg`, `dog.0.jpg`).

### 2. Data Preprocessing  
- Resize images to a consistent shape (e.g., 150x150 pixels).  
- Extract labels from filenames and convert them to numerical format (0 for cats, 1 for dogs).  
- Scale pixel values to a 0-1 range by dividing by 255.  
- Split dataset into training, validation, and testing subsets for robust evaluation.

### 3. CNN Model Architecture  
The model consists of the following layers:  

- **Convolutional Layers:** Multiple filters to detect edges, textures, and patterns.  
- **Pooling Layers:** Max pooling to reduce spatial dimensions and computation.  
- **Flatten Layer:** Converts 2D feature maps into a 1D vector.  
- **Dense (Fully Connected) Layers:** Learn non-linear feature combinations for classification.  
- **Dropout (optional):** Prevent overfitting by randomly disabling neurons during training.  
- **Output Layer:** Sigmoid activation for binary class probability.

### 4. Training  
- Compile the model with optimizer Adam and binary crossentropy loss.  
- Train over multiple epochs, monitoring accuracy and loss on validation data to prevent overfitting.

### 5. Evaluation  
- Assess performance on the test set by computing accuracy and loss.  
- Visualize training and validation accuracy and loss metrics over epochs to understand learning behavior.

---

## Code Analysis

| `main.py`                                   | `model.py`                                     |
|---------------------------------------------|------------------------------------------------|
| Data loading and preprocessing               | Defines and compiles CNN architecture           |
| Training and validation control               | Encapsulates model structure for easy changes   |
| Performance monitoring and logging            | Modular design aids experimentation              |

This separation enables clean, maintainable, and scalable code.

---

## Results and Observations

- Achieves **high accuracy (≈90%+)** in distinguishing dogs from cats.  
- Challenges include overfitting and ambiguous images, mitigated via dropout and validation.  
- GPU acceleration recommended for faster training on large datasets.

---

## Summary

1. Downloaded and inspected the Dogs vs Cats dataset.  
2. Preprocessed data: resized, scaled, and labeled images.  
3. Defined CNN architecture in `model.py`.  
4. Managed training and validation via `main.py`.  
5. Evaluated model with meaningful metrics and visualizations.  
6. Created a modular and extensible codebase for future improvements.

---

## Educational Value

- Understand core CNN concepts and applications in image classification.  
- Gain hands-on experience with data engineering, model building, and evaluation.  
- Learn the importance of modular codebases in machine learning projects.

---

## References

- Kaggle Dataset: [Dogs vs Cats](https://www.kaggle.com/datasets/salader/dogs-vs-cats)  
- GitHub Repository: [Source code](https://github.com/YatinG021/-image-Classification-model-using-CNN-/blob/main/README.md)
## OUTPUT
## Model Summary: Layer-Wise Architecture Overview

_Presents the complete architecture of the Convolutional Neural Network used for dog vs cat classification, detailing each layer type, output shape, and parameter count._

<img width="803" height="637" alt="Image" src="https://github.com/user-attachments/assets/19151e4d-3a6a-4bf6-a280-49aede2ff758" />

## Training and Validation Accuracy: Well-Generalized Model

*Shows both training and validation accuracy increasing over epochs with close performance, indicating a well-trained and generalized model without signs of overfitting.*

<img width="569" height="405" alt="Image" src="https://github.com/user-attachments/assets/35632aed-19c5-469e-b1ab-4b3e0a4d4a15" />


## Training and Validation Loss: Well-Generalized Model

*Depicts the loss decreasing for both training and validation sets at similar rates, confirming effective learning and good generalization.*

<img width="559" height="401" alt="Image" src="https://github.com/user-attachments/assets/a29bff7c-f263-4256-b998-58b6dd82c2db" />

## Training and Validation Accuracy: Overfitting Example

*Illustrates overfitting: the training accuracy continues to rise while the validation accuracy plateaus or decreases, showing the model learns training data too well but fails to generalize.*

<img width="560" height="408" alt="Image" src="https://github.com/user-attachments/assets/469397c7-2238-4514-bc7e-601b61c12361" />


## Training and Validation Loss: Overfitting Example
*Demonstrates overfitting by showing training loss decreasing steadily while validation loss begins to rise, indicating deteriorating performance on unseen data.*

<img width="553" height="405" alt="Image" src="https://github.com/user-attachments/assets/907d6dec-769b-4bbe-8200-e8783bd444d4" />



## Sample Output: Dog Image - Model Prediction Visualization

_Shows a sample input image representing a dog, displayed as processed by the CNN model during evaluation or inference._

<img width="638" height="447" alt="Image" src="https://github.com/user-attachments/assets/4c59fb7c-485f-46a0-9138-dd9bcec0e9bd" />


## Sample Output: Cat Image - Model Prediction Visualization

_Displays a sample input image representing a cat, as processed by the model during testing or inference._

<img width="628" height="406" alt="Image" src="https://github.com/user-attachments/assets/56a77934-dc24-4dcf-8425-84d8c026ab76" />

