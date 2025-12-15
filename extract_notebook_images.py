"""
Script to extract image outputs from Jupyter notebook cells and save them to images/ folder
"""
import json
import base64
import os

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

# Load the notebook
with open('retinalDiseaseClassifier_latest.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Image naming based on cell content/position
image_names = {
    14: 'diseases_per_sample_histogram',
    16: 'disease_comparison_0_vs_5',
    18: 'disease_samples_0_to_5',
    19: 'class_counts_histogram',
    26: 'transform_comparison',
    30: 'sample_transformed_image',
    39: 'lr_finder',
    52: 'model_comparison_imbalance_fix',
    54: 'confusion_matrices_rare_classes_v2',
    55: 'confusion_matrices_top10_v2',
    77: 'metrics_by_class',
    80: 'training_curves',
    82: 'confusion_matrices_top10',
    83: 'confusion_matrix_all_classes',
    84: 'confusion_matrices_all_classes_grid',
    85: 'precision_recall_f1_top15',
    86: 'roc_curves_top5',
    88: 'metrics_heatmap_all_classes',
    89: 'sample_predictions',
    90: 'class_distribution_vs_performance',
    91: 'overall_performance_pie',
}

saved_count = 0

for cell_idx, cell in enumerate(notebook['cells'], start=1):
    if cell['cell_type'] == 'code' and 'outputs' in cell:
        for output in cell['outputs']:
            # Check for image data
            if 'data' in output:
                if 'image/png' in output['data']:
                    # Get base64 image data
                    img_data = output['data']['image/png']
                    
                    # Determine filename
                    if cell_idx in image_names:
                        filename = f"images/{image_names[cell_idx]}.png"
                    else:
                        filename = f"images/cell_{cell_idx}_output.png"
                    
                    # Decode and save
                    img_bytes = base64.b64decode(img_data)
                    with open(filename, 'wb') as img_file:
                        img_file.write(img_bytes)
                    
                    print(f"✅ Saved: {filename}")
                    saved_count += 1

print(f"\n📁 Total images saved: {saved_count}")
print("📂 Images saved to: images/")
