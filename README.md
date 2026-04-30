🧠 Industrial Anomaly Detection System for Capsules

📌 Overview

This project presents an **unsupervised anomaly detection system** for pharmaceutical capsules using **Deep Learning and Computer Vision**.

The system detects surface defects such as:
- Crack  
- Scratch  
- Squeeze  
- Faulty Imprint  

It uses a **Convolutional Autoencoder** trained only on **normal (good) capsule images** and identifies anomalies based on **reconstruction error**.


🚀 Features

- Unsupervised anomaly detection  
- CNN-based Autoencoder  
- Image-level classification (GOOD / DEFECT)  
- Pixel-level anomaly localization  
- Heatmap visualization  
- End-to-end pipeline (Train → Test → Predict)  


🧠 How It Works

1. Train model on defect-free images  
2. Model learns normal capsule patterns  
3. Input image is reconstructed  
4. Reconstruction error is calculated  
5. If error > threshold → DEFECT  
6. Else → GOOD  


🛠️ Technologies Used

- Python  
- PyTorch  
- OpenCV  
- NumPy  
- Matplotlib  
- Computer Vision  
- Deep Learning  


📂 Project Structure
capsule_anomaly_detection/
│
├── dataset.py
├── train.py
├── test.py
├── predict.py
├── utils.py
├── requirements.txt
│
├── models/
   └── autoencoder.py


⚙️ Installation

bash
pip install -r requirements.txt

🔹 Train the Model
python train.py

🔹 Test the Model
python test.py

🔹 Predict Custom Image
python predict.py path_to_image


📊 Output
Prediction: GOOD / DEFECT
Anomaly Score
Heatmap Visualization


<img width="1264" height="699" alt="image" src="https://github.com/user-attachments/assets/02d5dabe-4efd-4f01-a316-e06c9fd70e0d" />

<img width="1594" height="700" alt="image" src="https://github.com/user-attachments/assets/99fc6153-0057-4642-8039-3a78f0e4fbb6" />

<img width="1247" height="664" alt="image" src="https://github.com/user-attachments/assets/835d9d9a-32fb-40e2-ab73-12d9222ab2c8" />


📈 Evaluation Metrics
ROC-AUC
Precision
Recall
F1-score
Confusion Matrix


🔮 Future Improvements
Use advanced models like PatchCore or GAN
Improve threshold selection
Real-time industrial deployment
Build web-based interface


👨‍💻 Author
Vraj Sangani
