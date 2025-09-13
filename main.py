# ============================================================================
# HOT DOG OR NOT HOT DOG CLASSIFIER
# A computer vision model to classify images as hot dogs or not hot dogs
# ============================================================================

# ============================================================================
# 1. SETUP AND INSTALLATION
# ============================================================================

# Install required packages
!pip install fastai
!pip install kaggle

# Import necessary libraries
import json
import os
from fastai.vision.all import *
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# 2. DATA DOWNLOAD AND SETUP
# ============================================================================

# Configure Kaggle API credentials
# NOTE: Replace with your actual Kaggle username and API key
kaggle_credentials = {
    "username": "YOUR_USERNAME",
    "key": "YOUR_KEY"
}

# Create Kaggle config directory and save credentials
kaggle_config_dir = '/root/.kaggle'
os.makedirs(kaggle_config_dir, exist_ok=True)
with open(f'{kaggle_config_dir}/kaggle.json', 'w') as credential_file:
    json.dump(kaggle_credentials, credential_file)

# Download and extract the hot dog dataset from Kaggle
!kaggle datasets download -d dansbecker/hot-dog-not-hot-dog
!unzip -q hot-dog-not-hot-dog.zip

# ============================================================================
# 3. DATA EXPLORATION
# ============================================================================

# Define paths to training and test data
training_data_path = Path('/content/train')
testing_data_path = Path('/content/test')

# Get lists of image files for each category
training_hotdog_images = get_image_files(training_data_path / 'hot_dog')
training_not_hotdog_images = get_image_files(training_data_path / 'not_hot_dog')
testing_hotdog_images = get_image_files(testing_data_path / 'hot_dog')
testing_not_hotdog_images = get_image_files(testing_data_path / 'not_hot_dog')

print(f"Training hot dog images: {len(training_hotdog_images)}")
print(f"Training not hot dog images: {len(training_not_hotdog_images)}")
print(f"Testing hot dog images: {len(testing_hotdog_images)}")
print(f"Testing not hot dog images: {len(testing_not_hotdog_images)}")

# Display sample hot dog image
sample_hotdog_image = Image.open(training_hotdog_images[0])
sample_hotdog_image.to_thumb(256, 256)

# Display sample not hot dog image
sample_not_hotdog_image = Image.open(training_not_hotdog_images[5])
sample_not_hotdog_image.to_thumb(256, 256)

# ============================================================================
# 4. DATA LOADING AND PREPROCESSING
# ============================================================================

# Create data loaders with preprocessing transformations
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
IMAGE_SIZE = 192
RANDOM_SEED = 42

data_loaders = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=VALIDATION_SPLIT, seed=RANDOM_SEED),
    get_y=parent_label,  # Use folder name as label
    item_tfms=[Resize(IMAGE_SIZE, method='squish')],  # Resize images
).dataloaders(training_data_path, bs=BATCH_SIZE)

# Display a batch of training images with labels
data_loaders.show_batch(max_n=6)

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================

# Create a convolutional neural network using ResNet-18 architecture
TRAINING_EPOCHS = 4
model = vision_learner(data_loaders, resnet18, metrics=error_rate)

# Fine-tune the pre-trained model on our hot dog dataset
print(f"Training model for {TRAINING_EPOCHS} epochs...")
model.fine_tune(TRAINING_EPOCHS)

# ============================================================================
# 6. RESULTS AND EVALUATION
# ============================================================================

# Display predictions on validation set
model.show_results(max_n=12, figsize=(15, 10))