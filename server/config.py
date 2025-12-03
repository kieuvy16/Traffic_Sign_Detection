"""
Server Configuration and Utilities
"""

import yaml
import logging
from pathlib import Path
import os

def load_config(config_path="config/server_config.yaml"):
    """
    Load server configuration from YAML file
    
    Args:
        config_path: Path to config file
    
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set default values if not specified
        config.setdefault('server', {})
        config['server'].setdefault('host', '0.0.0.0')
        config['server'].setdefault('port', 8000)
        config['server'].setdefault('debug', False)
        config['server'].setdefault('reload', False)
        
        config.setdefault('models', {})
        config['models'].setdefault('yolo', {})
        config['models']['yolo'].setdefault('path', 'models/yolov8/best.pt')
        config['models']['yolo'].setdefault('confidence', 0.5)
        config['models']['yolo'].setdefault('iou_threshold', 0.45)
        
        config['models'].setdefault('cnn', {})
        config['models']['cnn'].setdefault('path', 'models/cnn/classifier.h5')
        config['models']['cnn'].setdefault('input_size', [224, 224])
        
        config.setdefault('processing', {})
        config['processing'].setdefault('max_image_size', 5242880)  # 5MB
        config['processing'].setdefault('allowed_extensions', [".jpg", ".jpeg", ".png"])
        config['processing'].setdefault('output_format', "json")
        
        config.setdefault('cors', {})
        config['cors'].setdefault('allow_origins', ["*"])
        config['cors'].setdefault('allow_methods', ["*"])
        config['cors'].setdefault('allow_headers', ["*"])
        
        # Load YOLO classes from yolo config
        yolo_config_path = Path("config/yolo_config.yaml")
        if yolo_config_path.exists():
            with open(yolo_config_path, 'r') as f:
                yolo_config = yaml.safe_load(f)
                if 'classes' in yolo_config and 'names' in yolo_config['classes']:
                    config['classes'] = yolo_config['classes']
        
        return config
        
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        raise

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Path to log file (optional)
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )

def validate_config(config):
    """
    Validate configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        tuple: (is_valid, errors)
    """
    errors = []
    
    # Check required paths
    required_paths = [
        ('models.yolo.path', 'YOLO model path'),
        ('models.cnn.path', 'CNN model path')
    ]
    
    for path_key, description in required_paths:
        path = get_nested(config, path_key.split('.'))
        if not path:
            errors.append(f"{description} is required")
        elif not os.path.exists(path):
            errors.append(f"{description} does not exist: {path}")
    
    # Check server settings
    if not isinstance(config.get('server', {}).get('port'), int):
        errors.append("Server port must be an integer")
    
    if config.get('server', {}).get('port') not in range(1, 65536):
        errors.append("Server port must be between 1 and 65535")
    
    return len(errors) == 0, errors

def get_nested(dictionary, keys, default=None):
    """
    Get nested value from dictionary
    
    Args:
        dictionary: Source dictionary
        keys: List of keys
        default: Default value if key not found
    
    Returns:
        Value or default
    """
    for key in keys:
        if isinstance(dictionary, dict) and key in dictionary:
            dictionary = dictionary[key]
        else:
            return default
    return dictionary

def create_default_config(config_path="config/server_config.yaml"):
    """
    Create default configuration file
    
    Args:
        config_path: Path to save config file
    """
    default_config = {
        'server': {
            'host': '0.0.0.0',
            'port': 8000,
            'debug': False,
            'reload': False
        },
        'models': {
            'yolo': {
                'path': 'models/yolov8/best.pt',
                'confidence': 0.5,
                'iou_threshold': 0.45
            },
            'cnn': {
                'path': 'models/cnn/classifier.h5',
                'input_size': [224, 224]
            }
        },
        'processing': {
            'max_image_size': 5242880,
            'allowed_extensions': ['.jpg', '.jpeg', '.png'],
            'output_format': 'json'
        },
        'cors': {
            'allow_origins': ['*'],
            'allow_methods': ['*'],
            'allow_headers': ['*']
        }
    }
    
    # Create config directory if it doesn't exist
    config_dir = os.path.dirname(config_path)
    if config_dir and not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    return default_config

# Global config instance
_config = None

def get_config():
    """
    Get global configuration instance
    
    Returns:
        dict: Configuration dictionary
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config

if __name__ == "__main__":
    # Test configuration loading
    try:
        config = load_config()
        print("Configuration loaded successfully:")
        print(yaml.dump(config, default_flow_style=False))
        
        # Validate config
        is_valid, errors = validate_config(config)
        if is_valid:
            print("Configuration is valid")
        else:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
                
    except Exception as e:
        print(f"Error: {e}")