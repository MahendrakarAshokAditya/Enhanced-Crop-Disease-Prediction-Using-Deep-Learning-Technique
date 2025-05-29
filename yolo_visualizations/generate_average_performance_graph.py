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

# Calculate average performance metrics
avg_precision = results['metrics/precision(B)'].mean()
avg_recall = results['metrics/recall(B)'].mean()

# Calculate F1 score (harmonic mean of precision and recall)
# Using the formula: F1 = 2 * (precision * recall) / (precision + recall)
avg_f1_score = 2 * (results['metrics/precision(B)'] * results['metrics/recall(B)']) / \
              (results['metrics/precision(B)'] + results['metrics/recall(B)'])
avg_f1_score = avg_f1_score.mean()

# Calculate average mAP values
avg_map50 = results['metrics/mAP50(B)'].mean()
avg_map5095 = results['metrics/mAP50-95(B)'].mean()

# Get final metrics for comparison
final_metrics = results.iloc[-1]
final_precision = final_metrics['metrics/precision(B)']
final_recall = final_metrics['metrics/recall(B)']
final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall)
final_map50 = final_metrics['metrics/mAP50(B)']
final_map5095 = final_metrics['metrics/mAP50-95(B)']

# Create a figure for the average performance bar graph
plt.figure(figsize=(12, 8))

# Define metrics and their values
metrics = ['Precision', 'Recall', 'F1 Score', 'mAP@0.5', 'mAP@0.5:0.95']
avg_values = [avg_precision, avg_recall, avg_f1_score, avg_map50, avg_map5095]
final_values = [final_precision, final_recall, final_f1, final_map50, final_map5095]

# Set up bar positions
x = np.arange(len(metrics))
width = 0.35

# Create bars
plt.bar(x - width/2, avg_values, width, label='Average Across Training')
plt.bar(x + width/2, final_values, width, label='Final Model Performance')

# Add labels and title
plt.xlabel('Performance Metrics')
plt.ylabel('Score')
plt.title('YOLOv8 Average vs Final Performance for Crop Disease Detection')
plt.xticks(x, metrics)
plt.ylim(0, 1.0)

# Add value labels on top of each bar
for i, v in enumerate(avg_values):
    plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center')

for i, v in enumerate(final_values):
    plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center')

# Add legend
plt.legend()

# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add a text box with summary information
summary_text = f"Training Summary:\n"
summary_text += f"Total Epochs: {int(final_metrics['epoch'])}\n"
summary_text += f"Average Precision: {avg_precision:.3f} | Final: {final_precision:.3f}\n"
summary_text += f"Average Recall: {avg_recall:.3f} | Final: {final_recall:.3f}\n"
summary_text += f"Average F1 Score: {avg_f1_score:.3f} | Final: {final_f1:.3f}\n"
summary_text += f"Average mAP@0.5: {avg_map50:.3f} | Final: {final_map50:.3f}\n"
summary_text += f"Average mAP@0.5:0.95: {avg_map5095:.3f} | Final: {final_map5095:.3f}"

plt.figtext(0.5, 0.01, summary_text, ha='center', bbox=dict(facecolor='white', alpha=0.8), fontsize=10)

# Adjust layout
plt.tight_layout(rect=[0, 0.1, 1, 0.95])

# Save the figure
output_path = 'c:\\Users\\Dell\\OneDrive\\Desktop\\Enhancing_crop_disease_detection\\yolo_visualizations\\average_performance_graph.svg'
plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
print(f"Average performance graph saved to {output_path}")

# Create a second bar graph showing class-wise performance
plt.figure(figsize=(14, 8))

# Define metrics for class-wise comparison
class_metrics = ['Precision', 'Recall', 'F1 Score']
class_values = [
    [final_precision, final_recall, final_f1],  # Final model values
]

# Set up bar positions for class-wise graph
x = np.arange(len(class_metrics))
width = 0.25

# Create bars for class-wise comparison
plt.bar(x, class_values[0], width, label='All Classes Average')

# Add labels and title for class-wise graph
plt.xlabel('Performance Metrics')
plt.ylabel('Score')
plt.title('YOLOv8 Performance Metrics for Crop Disease Detection')
plt.xticks(x, class_metrics)
plt.ylim(0, 1.0)

# Add value labels on top of each bar for class-wise graph
for i, v in enumerate(class_values[0]):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center')

# Add legend for class-wise graph
plt.legend()

# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the class-wise figure
class_output_path = 'c:\\Users\\Dell\\OneDrive\\Desktop\\Enhancing_crop_disease_detection\\yolo_visualizations\\class_performance_graph.svg'
plt.savefig(class_output_path, format='svg', dpi=300, bbox_inches='tight')
print(f"Class performance graph saved to {class_output_path}")

# Show the plots
plt.show()