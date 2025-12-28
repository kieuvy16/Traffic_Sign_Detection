<<<<<<< HEAD
# 🚦 Hệ Thống Nhận Diện Biển Báo Giao Thông

Hệ thống nhận diện và phân loại biển báo giao thông sử dụng YOLOv8 cho object detection và CNN cho classification.

## 📋 Tính năng

- ✅ Phát hiện biển báo trong ảnh sử dụng YOLOv8
- ✅ Phân loại biển báo sử dụng CNN
- ✅ Web interface trực quan
- ✅ REST API cho integration
- ✅ Hỗ trợ batch processing
- ✅ Metrics và visualization
- ✅ Docker support

## 🏗️ Cấu trúc Project

```
traffic-sign-detection/
├── config/                    # File cấu hình
│   ├── server_config.yaml    # Cấu hình server
│   └── yolo_config.yaml      # Cấu hình YOLO training
├── data/                      # Dữ liệu
│   ├── raw/                  # Dữ liệu gốc
│   ├── processed/            # Dữ liệu đã xử lý
│   └── annotations/          # Annotations
├── models/                    # Models
│   ├── yolo/                 # YOLOv8 models
│   └── cnn/                  # CNN models
├── src/                       # Source code
│   ├── yolo_detector.py      # YOLO detector
│   ├── cnn_classifier.py     # CNN classifier
│   ├── data_preprocessing.py # Data preprocessing
│   ├── training/             # Training scripts
│   │   ├── train_yolo.py
│   │   └── train_cnn.py
│   └── utils/                # Utilities
│       ├── metrics.py
│       └── visualization.py
├── server/                    # Web server
│   ├── app.py                # FastAPI application
│   ├── routes.py             # API routes
│   └── config.py             # Server config
├── templates/                 # HTML templates
│   └── index.html
├── static/                    # Static files
│   ├── css/
│   ├── js/
│   └── images/
├── tests/                     # Tests
│   ├── test_detector.py
│   └── test_classifier.py
├── notebooks/                 # Jupyter notebooks
├── results/                   # Results
│   ├── images/
│   ├── metrics/
│   └── reports/
├── requirements.txt           # Dependencies
├── setup.py                   # Setup script
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker compose
└── README.md                 # Documentation
```

## 🚀 Cài đặt

### Yêu cầu

- Python 3.8+
- CUDA (optional, cho GPU support)
- Docker (optional)

### Cài đặt thủ công

1. Clone repository:
```bash
git clone https://github.com/Joycee23/Traffic_Sign_Detection.git
cd traffic-sign-detection
```

2. Tạo virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Cài đặt dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

4. Cài đặt package:
```bash
pip install -e .
```

### Cài đặt với Docker

```bash
docker-compose up -d
```

### Setup Project (Sau khi clone)

Sau khi clone repository, chạy script setup để tạo cấu trúc thư mục:

```bash
python setup_project.py
```

Script này sẽ:
- Tạo các thư mục cần thiết
- Tạo file cấu hình mẫu
- Hướng dẫn các bước tiếp theo

## 📊 Chuẩn bị dữ liệu

### Cấu trúc dữ liệu

Dữ liệu raw cần có cấu trúc:

```
data/raw/
├── images/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── labels/
    ├── image001.txt
    ├── image002.txt
    └── ...
```

**Lưu ý quan trọng**: Dataset được lưu trong thư mục `data/raw/archive/` và không được đẩy lên Git do kích thước lớn. Người dùng cần tự tải dataset và đặt vào đúng cấu trúc.

### Tải dataset

1. Tải dataset từ các nguồn công khai:
   - [Vietnam Traffic Signs](https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs)
   - [Roboflow Dataset](https://universe.roboflow.com/truong-a6rzc/bien-bao-giao-thong-viet-nam-zalo1/dataset/5)

2. Đặt dataset vào thư mục `data/raw/archive/` với cấu trúc:
   ```
   data/raw/archive/
   ├── images/
   ├── labels/
   ├── classes.txt
   ├── classes_en.txt
   └── classes_vie.txt
   ```

### Preprocessing

```bash
python src/data_preprocessing.py
```

Script này sẽ:
- Chia dataset thành train/val/test (70%/15%/15%)
- Tạo file data.yaml cho YOLO
- Áp dụng augmentation (optional)
- Tạo thư mục processed với cấu trúc YOLO

## 🎯 Training

### Train YOLO Model

```bash
python src/training/train_yolo.py
```

Cấu hình training trong `config/yolo_config.yaml`

### Train CNN Model

```bash
python src/training/train_cnn.py
```

## 🌐 Chạy Web Server

### Development mode

```bash
cd server
python app.py
```

hoặc

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Production mode

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4
```

Truy cập: http://localhost:8000

## 📡 API Endpoints

### Health Check
```
GET /health
```

### Detect Traffic Signs
```
POST /api/detect
Content-Type: multipart/form-data
Body: file (image)
```

### Batch Detection
```
POST /api/detect_batch
Content-Type: multipart/form-data
Body: files[] (multiple images)
```

### Classify Sign
```
POST /api/classify
Content-Type: multipart/form-data
Body: file (cropped sign image)
```

### Get Classes
```
GET /api/classes
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_detector.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📈 Các lớp biển báo

Hệ thống hỗ trợ 15 loại biển báo:

0. speed_limit_20
1. speed_limit_30
2. speed_limit_50
3. speed_limit_60
4. speed_limit_70
5. speed_limit_80
6. no_overtaking
7. no_entry
8. danger
9. mandatory_left
10. mandatory_right
11. mandatory_straight
12. stop
13. yield
14. priority_road

## 🔧 Cấu hình

### Server Configuration (config/server_config.yaml)

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  debug: false

models:
  yolo:
    path: "models/yolo11/train/weights/best.pt"
    confidence: 0.5
  cnn:
    path: "models/cnn/classifier.h5"
```

### YOLO Configuration (config/yolo_config.yaml)

```yaml
training:
  epochs: 100
  batch_size: 8
  img_size: 640
  device: "cuda"
```

## 📊 Metrics và Visualization

```python
from src.utils.metrics import MetricsCalculator
from src.utils.visualization import Visualizer

# Calculate metrics
metrics = MetricsCalculator.calculate_metrics(y_true, y_pred, class_names)

# Visualize results
visualizer = Visualizer()
visualizer.plot_confusion_matrix(y_true, y_pred, class_names)
visualizer.plot_training_history(history)
```

## 🐳 Docker

### Build image

```bash
docker build -t traffic-sign-detection .
```

### Run container

```bash
docker run -p 8000:8000 traffic-sign-detection
```

### Docker Compose

```bash
docker-compose up -d
```

## 📝 Logging

Logs được lưu trong thư mục `logs/`:
- Training logs
- Inference logs
- API access logs

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Jerry Nguyễn - Initial work

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- TensorFlow
- FastAPI
- OpenCV

## 📞 Contact

- Email: tuannguyen211982@gmail.com
- Project Link: https://github.com/Joycee23/Traffic_Sign_Detection
## 🔄 Updates

### Version 1.0.0
- Initial release
- YOLO detection
- CNN classification
- Web interface
- REST API

## 📚 References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
=======
# Traffic_Sign_Detection
 8a1d2e40b782ed1556d730ac04207d48c659bc2f
