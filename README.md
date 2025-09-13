# Hot Dog or Not Hot Dog 🌭

> "I can only do hot dog, not hot dog. Nothing else."

This recreates Jian-Yang's "Not Hotdog" app from [Silicon Valley Season 4](https://www.youtube.com/watch?v=tWwCK95X6go), where the character builds a simple binary classifier that can only identify hot dogs.



## How It Works

It uses a fine-tuned ResNet-18 convolutional neural network built with FastAI to distinguish between hot dog and not hot dog images.

**Architecture:**
- Pre-trained ResNet-18 model
- Transfer learning with 4 epochs of fine-tuning
- Binary classification (hot dog vs not hot dog)
- 192x192 pixel image input

## Results

![Classifier Results](hot_dog_classifier_results.jpeg)
