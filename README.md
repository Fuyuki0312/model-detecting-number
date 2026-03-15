## Single Handwritten Numerical Digit Classification
Image classification using Convolutional Neural Network, demonstrating inference pipeline with PyTorch.


## Project Overview
- Task: Numerical Image Classification (from 0 to 9)
- Model: Convolutional Neural Network (CNN)
- Test Accuracy: 99.9%


## Dataset
- The dataset is from Kaggle and is not included due to size limitations.

- Number of images: approximately 25000
- Image size: 90x140

- Structure:
- numbers/
- ├─ 0/
- ├─ 1/
- ├─ 2/
- ├─ ...
- ├─ 9/


## Demonstration
- A demonstration of the model is produced at: https://huggingface.co/spaces/Fuyuki0312/ModelDetectingNumber-demo
- Note: Input images are converted to grayscale and resized before inference and their background color should be white.


## Limitation
- Model usually gives right predictions only when the background color of input images is white because this model was trained primarily on numerical images with white backgrounds.
- If the input image is not so clear, the model may confidently produce a wrong prediction.


## Possible Improvements
- Expanding the dataset to include numerical images with diverse backgrounds (dark, textured, etc).
- Applying background-related data augmentation techniques during training.


## Others
- This project was built in early 2026 as part of my journey to become an AI Engineer.
