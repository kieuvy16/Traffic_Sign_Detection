
import os
import cv2
import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

class DataPreprocessor:
    def __init__(self, config_path="config/yolo_config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
    def prepare_yolo_dataset(self, raw_data_path, output_path):
        """
        Chuẩn bị dataset cho YOLOv8
        Giả sử dataset có cấu trúc:
        raw_data_path/
            ├── images/
            └── labels/
        """
        print("Preparing YOLO dataset...")
        
        images_path = Path(raw_data_path) / "images"
        labels_path = Path(raw_data_path) / "labels"
        
        # Lấy tất cả các file ảnh
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        
        # Chia dataset
        train_imgs, temp_imgs = train_test_split(image_files, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
        
        # Copy files vào các thư mục tương ứng
        for split, img_list in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            split_img_dir = Path(output_path) / split / "images"
            split_label_dir = Path(output_path) / split / "labels"
            
            split_img_dir.mkdir(parents=True, exist_ok=True)
            split_label_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in tqdm(img_list, desc=f"Processing {split}"):
                # Copy image
                shutil.copy(img_path, split_img_dir / img_path.name)
                
                # Copy label
                label_file = labels_path / f"{img_path.stem}.txt"
                if label_file.exists():
                    shutil.copy(label_file, split_label_dir / label_file.name)
        
        print(f"Dataset prepared: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")
        
        # Tạo file data.yaml cho YOLO
        self._create_data_yaml(output_path)
    
    def _create_data_yaml(self, output_path):
        """Tạo file data.yaml cho YOLOv8"""
        data_yaml = {
            'path': str(Path(output_path).absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.config['classes']['names']),
            'names': self.config['classes']['names']
        }
        
        with open(Path(output_path) / "data.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(data_yaml, f, allow_unicode=True)
        
        print(f"Created data.yaml at {output_path}/data.yaml")
    
    def augment_images(self, image_path, save_path, num_augmentations=5):
        """Augmentation cho ảnh"""
        import albumentations as A
        
        transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.HueSaturationValue(p=0.5),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
        ])
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for i in range(num_augmentations):
            augmented = transform(image=image)['image']
            save_file = Path(save_path) / f"{Path(image_path).stem}_aug_{i}.jpg"
            cv2.imwrite(str(save_file), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.prepare_yolo_dataset("data/raw/archive", "data/processed")