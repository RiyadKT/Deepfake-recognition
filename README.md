# **Deepfake Recognition Project**

![Deepfake Detection Banner](images/deepfake_banner.jpeg)  


---

## **Overview**

This project aims to detect deepfake videos using video data from the **Centrale Supélec Deepfake Recognition Hackathon 2024** dataset, available on Kaggle. The solution leverages **PyTorch** for video classification and focuses on preprocessing, training, and evaluating a deep learning model.

---

## **Features**

- **Preprocessing**:  
  - Crops faces from video frames using **MTCNN (Multi-Task Cascaded Convolutional Networks)**.  
  - Resizes cropped frames to 224x224 pixels for model input.  

- **Model Architecture**:  
  - Uses a **pretrained MobileNetV2** model for binary classification (real vs. fake).  
  - Modifies the classifier layer for deepfake detection.  

- **Training and Optimization**:  
  - Implements a training loop with **Adam optimizer** and **CrossEntropy loss**.  
  - Balances the dataset to address class imbalance.  

- **Evaluation**:  
  - Generates a **confusion matrix** to visualize model performance.  
  - Computes accuracy on the validation set.

---

## **Dataset**

### **Video Sample**

Below is a sample video from the dataset:

[Watch Sample Video on YouTube](https://www.youtube.com/watch?v=h_4vV6asrMU)

- **Format**: MP4, 30 FPS, 720p resolution.  
- **Source**: [Kaggle Deepfake Detection Challenge](https://www.kaggle.com/competitions/deepfake-detection-challenge/data).  
- **Structure**: Videos labeled as either "real" or "fake" in `metadata.json`.  
- **Preprocessing**: Cropped face videos are stored in the `cropped_faces` directory.

---

## **Pipeline**

1. **Preprocessing**:  
   - Detect faces in video frames using **MTCNN**.  
   - Crop and resize faces.  
   - Save cropped faces as new videos.

2. **Dataset Preparation**:  
   - Parse `metadata.json` to extract labels.  
   - Balance the dataset (real vs. fake).  
   - Split into training and validation sets.

3. **Model Training**:  
   - Use **MobileNetV2** for binary classification.  
   - Train for 10 epochs with logging of loss and accuracy.

4. **Evaluation**:  
   - Compute predictions on the validation set.  
   - Generate a confusion matrix.

---

## **Model Architecture**

![Model Diagram](images/MobileNet.png)  

- **Base Model**: MobileNetV2 pretrained on ImageNet.  
- **Classifier**: Fully connected layer with two outputs (real or fake).  
- **Optimizer**: Adam optimizer with a learning rate of 0.001.  
- **Loss Function**: CrossEntropyLoss.

---

## **Results**

### **Confusion Matrix**
![Confusion Matrix](images/conf_matrix.png)  

- **Accuracy**: Achieved 87% accuracy on the validation set.  

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/deepfake-recognition.git
   cd deepfake-recognition
   ```

2. Install all required dependencies from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Project Structure**

```bash
Deepfake-recognition/
├── cropped_faces/                # Directory for cropped face videos
├── deepfake-detection-challenge/ # Dataset directory
├── images/                       # Directory for images (e.g., confusion matrix, banners)
├── Code.ipynb                    # Main notebook for the project
├── test_Code.ipynb               # Test notebook
├── deepfake_model.pth            # Saved model weights
├── metadata.json                 # Metadata for video labels
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
└── sample_submission.csv         # Sample submission file
```
