# Deteksi Bola

**Soal :**
Project uas deteksi objek dengan roboflow
1. tentukan objek yang akan di deteksi (cth : koin, piring, pensil)
2. siapkan minimal 80 gambar
3. model yang sudah di buat pada roboflow, gunakan pada google collab
4. tunjukkan hasil deteksi dengan gambar baru pada google collab

 # Jawaban

```python
# Langkah 1: Instalasi dan Import Library
!pip install ultralytics roboflow
from roboflow import Roboflow
from ultralytics import YOLO
from google.colab import files
from IPython.display import Image, display
from PIL import Image as PILImage
import os
import glob

# Langkah 2: Mengunduh Dataset dari Roboflow
rf = Roboflow(api_key="uuSym5k4bB2YSX3RkixW")
project = rf.workspace("uas-pengolahan-citra-digital").project("ball-auto-dataset")
version = project.version(1)
dataset = version.download("yolov8")
                

# Langkah 3: Memuat Model yang Sudah Dilatih
#model = YOLO("runs/detect/train/weights/best.pt")  # Ganti dengan path model yang sudah dilatih

model = YOLO("yolov8n.pt")  # Gunakan YOLOv8 versi Nano untuk kecepatan
#data_path = dataset.location + "/data.yaml" # Path file data.yaml yang disediakan oleh Roboflow

# Melatih model
#model.train(data=data_path, epochs=50, imgsz=640)

# Langkah 4: Input Gambar untuk Deteksi
print("Silakan unggah gambar yang ingin dideteksi:")
uploaded = files.upload()  # Mengunggah file gambar

# Deteksi dan simpan hasilnya
for file_name in uploaded.keys():
    print(f"Gambar berhasil diunggah: {file_name}")
    
    # Melakukan prediksi
    result = model.predict(source=file_name, save=True)
    
    # Langkah 5: Mencari folder terbaru yang berisi hasil deteksi
    result_dir_parent = "runs/detect"  # Direktori induk hasil deteksi
    result_dirs = glob.glob(os.path.join(result_dir_parent, "predict*"))  # Cari folder yang namanya diawali dengan 'predict'
    
    if result_dirs:
        # Menemukan folder terbaru berdasarkan waktu pembuatan
        latest_result_dir = max(result_dirs, key=os.path.getmtime)  # Menentukan folder yang paling baru
        print(f"Folder hasil deteksi terbaru: {latest_result_dir}")
        
        # Menemukan file gambar hasil deteksi dalam folder terbaru
        detected_files = glob.glob(os.path.join(latest_result_dir, "*"))  # Mengambil semua file di dalam folder
        detected_image_files = [f for f in detected_files if f.endswith(('.jpg', '.jpeg', '.png'))]  # Filter hanya gambar
        
        # Menampilkan gambar hasil deteksi dengan ukuran yang diperkecil
    if detected_image_files:
      output_image_path = detected_image_files[0]  # Ambil gambar pertama yang ditemukan
      print(f"Hasil deteksi disimpan di: {output_image_path}")

    img = PILImage.open(output_image_path)
    img_resized = img.resize((int(img.width * 0.5), int(img.height * 0.5)))  # Perkecil ukuran 50%
    
    display(img_resized)

```
**Output :** <br>
![download (2)](https://github.com/user-attachments/assets/056db701-a2c3-49fe-9d6b-05b64c7fe6a5)

