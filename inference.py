"""
Inference script for Retinal Disease Classifier.

Usage:
    python inference.py --image PATH_TO_IMAGE [--model MODEL_PATH]
    python inference.py --folder PATH_TO_FOLDER [--model MODEL_PATH]
    
Example:
    python inference.py --image ./test_image.png --model ./outputs/checkpoints/best_model.pth
"""

import argparse
import os
import sys
import glob
import torch
import pandas as pd
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src import model
from src import transforms


# Default class names (update if your model has different classes)
DEFAULT_CLASS_NAMES = [
    'DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN', 'ERM', 'LS', 'MS',
    'CSR', 'ODC', 'CRVO', 'TV', 'AH', 'ODP', 'ODE', 'ST', 'AION', 'PT',
    'RT', 'RS', 'CRS', 'EDN', 'RPEC', 'MHL', 'RP', 'CWS', 'CB', 'PRH',
    'CNV', 'VMT', 'RB', 'VH', 'CRAO', 'HTR', 'ASR', 'OTHER', 'TSNO',
    'MPED', 'RD', 'BRAO', 'LSCD'
]


class RetinalDiseasePredictor:
    """
    Predictor class for retinal disease classification.
    
    Args:
        model_path: Path to trained model checkpoint
        device: Device to use for inference
        threshold: Classification threshold
    """
    
    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        threshold: float = 0.5,
        class_names: list = None
    ):
        self.device = device or config.DEVICE
        self.threshold = threshold
        self.class_names = class_names or DEFAULT_CLASS_NAMES
        
        # Load model
        self.model = model.create_convnext_model(num_classes=len(self.class_names))
        
        if model_path and os.path.exists(model_path):
            self.model = model.load_checkpoint(self.model, model_path, self.device)
            print(f"✓ Model loaded from: {model_path}")
        else:
            print("⚠️ No model checkpoint provided. Using untrained model.")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get transforms
        self.transform = transforms.get_inference_transform()
    
    def predict_single(self, image_path: str) -> dict:
        """
        Predict diseases for a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with predictions
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(img_tensor)
            probabilities = torch.sigmoid(logits).cpu().squeeze()
        
        # Get predictions
        predictions = (probabilities > self.threshold).float()
        
        # Create result dictionary
        result = {
            'image_path': image_path,
            'predicted_diseases': [],
            'probabilities': {},
            'is_healthy': True
        }
        
        for i, (name, prob, pred) in enumerate(zip(
            self.class_names, probabilities.numpy(), predictions.numpy()
        )):
            result['probabilities'][name] = float(prob)
            if pred == 1:
                result['predicted_diseases'].append(name)
                result['is_healthy'] = False
        
        return result
    
    def predict_batch(self, image_paths: list) -> list:
        """
        Predict diseases for multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for path in image_paths:
            result = self.predict_single(path)
            results.append(result)
        return results
    
    def predict_folder(self, folder_path: str, pattern: str = '*.png') -> pd.DataFrame:
        """
        Predict diseases for all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            pattern: Glob pattern for image files
            
        Returns:
            DataFrame with predictions
        """
        image_paths = glob.glob(os.path.join(folder_path, pattern))
        
        if not image_paths:
            print(f"No images found in {folder_path} with pattern {pattern}")
            return pd.DataFrame()
        
        print(f"Found {len(image_paths)} images")
        
        results = self.predict_batch(image_paths)
        
        # Convert to DataFrame
        data = []
        for result in results:
            row = {
                'image': os.path.basename(result['image_path']),
                'is_healthy': result['is_healthy'],
                'diseases': ', '.join(result['predicted_diseases'])
            }
            # Add probabilities
            for name, prob in result['probabilities'].items():
                row[f'{name}_prob'] = prob
            data.append(row)
        
        return pd.DataFrame(data)


def print_prediction(result: dict):
    """Pretty print a prediction result."""
    print("\n" + "=" * 60)
    print(f"Image: {os.path.basename(result['image_path'])}")
    print("=" * 60)
    
    if result['is_healthy']:
        print("✅ Prediction: HEALTHY (No diseases detected)")
    else:
        print("⚠️ Prediction: DISEASES DETECTED")
        print("\nDetected diseases:")
        for disease in result['predicted_diseases']:
            prob = result['probabilities'][disease]
            print(f"  • {disease}: {prob:.1%}")
    
    # Show top 5 probabilities
    sorted_probs = sorted(
        result['probabilities'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:5]
    
    print("\nTop 5 probabilities:")
    for name, prob in sorted_probs:
        marker = "▓" * int(prob * 20) + "░" * (20 - int(prob * 20))
        print(f"  {name:8s} [{marker}] {prob:.1%}")
    
    print("=" * 60)


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Retinal Disease Inference')
    
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder of images')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output CSV path for folder predictions')
    parser.add_argument('--pattern', type=str, default='*.png',
                        help='Image file pattern for folder mode')
    
    args = parser.parse_args()
    
    if not args.image and not args.folder:
        parser.error("Must provide either --image or --folder")
    
    # Create predictor
    predictor = RetinalDiseasePredictor(
        model_path=args.model,
        threshold=args.threshold
    )
    
    # Single image prediction
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            return
        
        result = predictor.predict_single(args.image)
        print_prediction(result)
    
    # Folder prediction
    if args.folder:
        if not os.path.exists(args.folder):
            print(f"Error: Folder not found: {args.folder}")
            return
        
        df = predictor.predict_folder(args.folder, pattern=args.pattern)
        
        if not df.empty:
            df.to_csv(args.output, index=False)
            print(f"\n✅ Predictions saved to: {args.output}")
            
            # Summary
            healthy_count = df['is_healthy'].sum()
            disease_count = len(df) - healthy_count
            print(f"\nSummary:")
            print(f"  Total images: {len(df)}")
            print(f"  Healthy: {healthy_count}")
            print(f"  With diseases: {disease_count}")


if __name__ == '__main__':
    main()
