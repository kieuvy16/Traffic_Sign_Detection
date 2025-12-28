import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2

class CNNClassifier:
    def __init__(self, num_classes, input_shape=(224, 224, 3), use_sparse_labels=True):
        """Khởi tạo CNN classifier"""
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.use_sparse_labels = use_sparse_labels  # True nếu nhãn là số nguyên (0..C-1), False nếu one-hot
    
    def build_model(self):
        """Xây dựng mô hình CNN"""
        model = keras.Sequential([
            # Block 1
            layers.Conv2D(16, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Chọn loss phù hợp
        loss_fn = 'sparse_categorical_crossentropy' if self.use_sparse_labels else 'categorical_crossentropy'
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss_fn,
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, train_generator, val_generator, epochs=50, callbacks=None):
        """Train model"""
        if self.model is None:
            self.build_model()
        
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history
    
    def predict(self, image):
        """Dự đoán cho một ảnh"""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize và normalize
        image = cv2.resize(image, self.input_shape[:2])
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Predict
        predictions = self.model.predict(image, verbose=0)
        class_id = np.argmax(predictions[0])
        confidence = predictions[0][class_id]
        
        return {
            'class_id': int(class_id),
            'confidence': float(confidence),
            'all_probabilities': predictions[0].tolist()
        }
    
    def save(self, path):
        """Lưu model"""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model"""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")

if __name__ == "__main__":
    # Ví dụ: nếu dataset có 2 lớp và nhãn là số nguyên (0 hoặc 1)
    classifier = CNNClassifier(num_classes=2, use_sparse_labels=True)
    model = classifier.build_model()
    model.summary()