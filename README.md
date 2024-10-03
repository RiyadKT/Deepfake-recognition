# Deepfake Recognition Project

Overview

This project aims to detect deepfakes using video data from the Centrale Supélec Deepfake Recognition Hackathon 2024 dataset, available on Kaggle. The solution leverages a deep learning approach with a 3D ResNet-18 (r3d_18) model from PyTorch for video classification.

Features:
Preprocessing: The videos are preprocessed by cropping faces from each frame using the MTCNN (Multi-Task Cascaded Convolutional Networks). Cropped frames are resized to 112x112 pixels and converted to grayscale for input to the model.
Model Architecture: A pretrained 3D ResNet-18 model is used to extract temporal and spatial features from video sequences. The fully connected layer has been modified to classify videos as either real or fake.
Training and Optimization: The model is trained using the Adam optimizer with a CrossEntropy loss function. The training loop includes the transformation of video frames into tensors and batching using a custom PyTorch Dataset class.
Evaluation: After training, a confusion matrix is generated to visualize the performance of the model.
Data

The dataset consists of videos labeled as either "real" or "fake" in a provided metadata file (metadata.json). The videos have been preprocessed to crop the faces, and the cropped videos are stored in the cropped_videos directory.

