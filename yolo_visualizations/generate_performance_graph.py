import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set the style for the plots
plt.style.use('ggplot')

# Define the path to the results.csv file
results_path = 'c:\\Users\\Dell\\OneDrive\\Desktop\\Enhancing_crop_disease_detection\\runs\\runs\\detect\\train\\results.csv'

# Check if the file exists
if not os.path.exists(results_path):
    print(f"Error: Could not find results file at {results_path}")
    exit(1)

# Load the results data
print(f"Loading training results from {results_path}...")
results = pd.read_csv(results_path)

# Create a figure with multiple subplots
fig = plt.figure(figsize=(15, 12))
fig.suptitle('YOLOv8 Training Performance Metrics', fontsize=16)

# 1. Plot training and validation losses
ax1 = plt.subplot(2, 2, 1)
ax1.plot(results['epoch'], results['train/box_loss'], label='Train Box Loss')
ax1.plot(results['epoch'], results['train/cls_loss'], label='Train Class Loss')
ax1.plot(results['epoch'], results['train/dfl_loss'], label='Train DFL Loss')
ax1.plot(results['epoch'], results['val/box_loss'], '--', label='Val Box Loss')
ax1.plot(results['epoch'], results['val/cls_loss'], '--', label='Val Class Loss')
ax1.plot(results['epoch'], results['val/dfl_loss'], '--', label='Val DFL Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Losses')
ax1.legend()

# 2. Plot precision, recall, and mAP metrics
ax2 = plt.subplot(2, 2, 2)
ax2.plot(results['epoch'], results['metrics/precision(B)'], label='Precision')
ax2.plot(results['epoch'], results['metrics/recall(B)'], label='Recall')
ax2.plot(results['epoch'], results['metrics/mAP50(B)'], label='mAP@0.5')
ax2.plot(results['epoch'], results['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Metric Value')
ax2.set_title('Detection Performance Metrics')
ax2.legend()

# 3. Plot learning rate
ax3 = plt.subplot(2, 2, 3)
ax3.plot(results['epoch'], results['lr/pg0'])
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Learning Rate')
ax3.set_title('Learning Rate Schedule')

# 4. Plot combined loss trend
ax4 = plt.subplot(2, 2, 4)
total_train_loss = results['train/box_loss'] + results['train/cls_loss'] + results['train/dfl_loss']
total_val_loss = results['val/box_loss'] + results['val/cls_loss'] + results['val/dfl_loss']
ax4.plot(results['epoch'], total_train_loss, label='Total Train Loss')
ax4.plot(results['epoch'], total_val_loss, label='Total Val Loss')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Combined Loss')
ax4.set_title('Total Loss Trend')
ax4.legend()

# Add final metrics as text annotation
final_metrics = results.iloc[-1]
metrics_text = f"Final Metrics (Epoch {int(final_metrics['epoch'])}):\n"
metrics_text += f"Precision: {final_metrics['metrics/precision(B)']:.3f}\n"
metrics_text += f"Recall: {final_metrics['metrics/recall(B)']:.3f}\n"
metrics_text += f"mAP@0.5: {final_metrics['metrics/mAP50(B)']:.3f}\n"
metrics_text += f"mAP@0.5:0.95: {final_metrics['metrics/mAP50-95(B)']:.3f}\n"
metrics_text += f"Box Loss: {final_metrics['val/box_loss']:.3f}\n"
metrics_text += f"Class Loss: {final_metrics['val/cls_loss']:.3f}\n"
metrics_text += f"DFL Loss: {final_metrics['val/dfl_loss']:.3f}"

fig.text(0.5, 0.01, metrics_text, ha='center', bbox=dict(facecolor='white', alpha=0.5))

# Adjust layout
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# Save the figure
output_path = 'yolo_performance_analysis.svg'
plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
print(f"Performance graph saved to {output_path}")

# Show the plot
plt.show()

# Generate a summary report
print("\nGenerating performance summary...")

# Calculate improvement metrics
first_epoch = results.iloc[0]
improvement_precision = final_metrics['metrics/precision(B)'] - first_epoch['metrics/precision(B)']
improvement_recall = final_metrics['metrics/recall(B)'] - first_epoch['metrics/recall(B)']
improvement_map50 = final_metrics['metrics/mAP50(B)'] - first_epoch['metrics/mAP50(B)']
improvement_map5095 = final_metrics['metrics/mAP50-95(B)'] - first_epoch['metrics/mAP50-95(B)']

print(f"Training completed over {int(final_metrics['epoch'])} epochs")
print(f"Final Precision: {final_metrics['metrics/precision(B)']:.3f} (improved by {improvement_precision:.3f})")
print(f"Final Recall: {final_metrics['metrics/recall(B)']:.3f} (improved by {improvement_recall:.3f})")
print(f"Final mAP@0.5: {final_metrics['metrics/mAP50(B)']:.3f} (improved by {improvement_map50:.3f})")
print(f"Final mAP@0.5:0.95: {final_metrics['metrics/mAP50-95(B)']:.3f} (improved by {improvement_map5095:.3f})")