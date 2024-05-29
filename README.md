# Diabetic Retinopathy Image Classifier

![Diabetic Retinopathy Classifier](https://github.com/GOUTHAM183/Diabetic-Retinopathy-Classifier/blob/main/data/train/DR/0083ee8054ee_png.rf.1490d8387b7078fa60b8e4dfee77e4a5.jpg)


[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0-brightgreen)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8-blue)](https://www.python.org/)

## 📋 Project Overview

This project is a Convolutional Neural Network (CNN) based classifier built using TensorFlow to detect Diabetic Retinopathy (DR) from retinal images. The model achieved an impressive **94% accuracy** on the test dataset.

## 🌟 Features

- **High Accuracy**: Achieved 94% accuracy on the test set.
- **Data Augmentation**: Enhanced model generalization using various data augmentation techniques.
- **Transfer Learning**: Utilized pre-trained VGG16 model for better performance.

## 📂 Project Structure
```plaintext
diabetic-retinopathy-classifier/
├── data/
│   ├── train/
│   │   ├── DR/
│   │   └── No_DR/
│   ├── valid/
│   │   ├── DR/
│   │   └── No_DR/
│   └── test/
│       ├── DR/
│       └── No_DR/
├── notebooks/
│   └── Diabetic_Retinopathy_Classifier.ipynb
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Scikit-learn

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/GOUTHAM183/Diabetic-Retinopathy-Classifier.git
    cd diabetic-retinopathy-classifier
    ```

2. Install the required packages:

    ```bash
    pip install tensorflow numpy matplotlib scikit-learn
    ```


### Training the Model

To train the model, use the provided Jupyter notebook `Diabetic_Retinopathy_Classifier.ipynb` in the `notebooks` directory. This notebook includes all the steps for data loading, augmentation, model building, training, and evaluation.

### Evaluating the Model

Evaluate the model using the test set:

1. Load the trained model:
    ```python
    from tensorflow.keras.models import load_model

    model = load_model('models/diabetic_retinopathy_classifier.h5')
    ```

2. Evaluate the model:
    ```python
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
    print(f'Test accuracy: {test_acc:.2f}')
    ```




### Dataset

Place your dataset in the `data` directory

