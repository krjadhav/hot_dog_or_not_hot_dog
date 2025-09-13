# Hot Dog or Not Hot Dog 🌭

A computer vision classifier that determines whether an image contains a hot dog or not, inspired by the iconic app from HBO's Silicon Valley.

## Inspiration

This project recreates Jian-Yang's "Not Hotdog" app from [Silicon Valley Season 4](https://www.youtube.com/watch?v=tWwCK95X6go), where the character builds a simple binary classifier that can only identify hot dogs.

> "I can only do hot dog, not hot dog. Nothing else."

## How It Works

The classifier uses a fine-tuned ResNet-18 convolutional neural network built with FastAI to distinguish between hot dog and not hot dog images.

**Architecture:**
- Pre-trained ResNet-18 model
- Transfer learning with 4 epochs of fine-tuning
- Binary classification (hot dog vs not hot dog)
- 192x192 pixel image input

## Results

![Classifier Results](hot_dog_classifier_results.jpeg)

The model successfully identifies various hot dog presentations while correctly classifying other foods as "not hot dog," just like Jian-Yang's app.

## Usage

1. Install dependencies:
   ```bash
   pip install fastai kaggle
   ```

2. Configure Kaggle credentials in `main.py`

3. Run the Jupyter notebook cells in `main.py`

The model will download the dataset, train on hot dog images, and display prediction results on test images.

## Dataset

Uses the "Hot Dog Not Hot Dog" dataset from Kaggle, containing thousands of labeled food images for training and validation.