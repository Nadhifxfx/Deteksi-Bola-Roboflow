# Deteksi Bola

**Soal :**
Project uas deteksi objek dengan roboflow
1. tentukan objek yang akan di deteksi (cth : koin, piring, pensil)
2. siapkan minimal 80 gambar
3. model yang sudah di buat pada roboflow, gunakan pada google collab
4. tunjukkan hasil deteksi dengan gambar baru pada google collab

 # Jawaban

```python
# Step 1: Install necessary libraries
!pip install ultralytics  # Install YOLOv8
!pip install matplotlib opencv-python-headless
!pip install roboflow

# Step 2: Import libraries
import matplotlib.pyplot as plt
from ultralytics import YOLO
from google.colab import files
import cv2
import numpy as np

from roboflow import Roboflow
rf = Roboflow(api_key="uuSym5k4bB2YSX3RkixW")
project = rf.workspace("uas-pengolahan-citra-digital").project("ball-auto-dataset")
version = project.version(1)
dataset = version.download("yolov8")

# Lihat folder tempat dataset diunduh
dataset_location = dataset.location  # dari RoboFlow download
print("Dataset downloaded to:", dataset_location)

from ultralytics import YOLO
# Buat model YOLOv8 baru
model = YOLO("yolov8n.pt")  # "yolov8n.pt" adalah versi YOLOv8 Nano

# Jalankan pelatihan dengan dataset
model.train(data="/content/Ball-auto-dataset-1/data.yaml", epochs=100, imgsz=320)

# Step 3: Upload image
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# Step 4: Perform object detection
results = model(image_path)  # Run inference on the uploaded image

# Step 5: Visualize results
# Save the annotated image
annotated_img = results[0].plot()
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

```
**Output :** <br>
![download (1)](https://github.com/user-attachments/assets/05575278-8b49-477c-ab21-edb61e8e284b)

