import matplotlib.pyplot as plt
import numpy as np
import os

# --- Data from the table ---
# Number of shots for the x-axis
shots = [1, 4, 16]

# Accuracy values for each method
# Note: Using np.nan for missing data points if any method wasn't tested at all shots.
methods = {
    'CoOp(4)': [72.7, 86.6, 96.4],
    'CoOp(16)': [78.3, 92.2, 96.8],
    'CoCoOp': [73.4, 81.5, 89.1],
    'CLIP-Adapter': [71.3, 73.1, 92.9],
    'CLIP-LoRA': [83.2, 93.7, 98.0],
    'CLIP-MoRE': [79.2, 93.5, 96.0]
}

# --- Plot Styling ---
# Use a modern and clean plot style
plt.style.use('seaborn-v0_8-whitegrid')

# Create a figure and axes for more control over the plot
fig, ax = plt.subplots(figsize=(10, 7))

# Define a color palette and marker styles for better visual distinction
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#D55E00']
markers = ['o', 's', '^', 'D', 'v', 'P']

# --- Plotting the data ---
for i, (name, accuracy) in enumerate(methods.items()):
    ax.plot(shots, accuracy, marker=markers[i], linestyle='-', linewidth=2, markersize=8, label=name, color=colors[i])

# --- Customizing the Plot Appearance ---
# Set titles and labels with appropriate font sizes for clarity
ax.set_title('Few-Shot Performance on Flowers-102', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Number of Shots', fontsize=12)
ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)

# Set x-axis to a logarithmic scale to better visualize performance at low shots,
# and ensure ticks appear for each shot count.
ax.set_xscale('log')
ax.set_xticks(shots)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.tick_params(axis='both', which='major', labelsize=10)


# Adjust y-axis to provide some space around the min/max values
min_acc = min(min(v) for v in methods.values())
max_acc = max(max(v) for v in methods.values())
ax.set_ylim(min_acc - 5, max_acc + 2)


# Place the legend outside the plot area for a cleaner look
ax.legend(title='Method', fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left')

# Remove top and right spines for a cleaner aesthetic
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add a light grid for the y-axis
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)


# --- Saving the Plot ---
# Define the output directory and filename
output_dir = 'graphics'
output_filename = 'flowers_plot.png'
output_path = os.path.join(output_dir, output_filename)

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Save the figure with high resolution and tight layout
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# Optionally, display the plot
plt.show()

print(f"Plot successfully saved to: {output_path}")

