# import matplotlib.pyplot as plt
# import numpy as np
# import os

# # --- Data from the table ---
# # Number of shots for the x-axis
# shots = [1, 4, 16]

# # Accuracy values for each method
# # Note: Using np.nan for missing data points if any method wasn't tested at all shots.
# methods = {
#     'CoOp(4)': [72.7, 86.6, 96.4],
#     'CoOp(16)': [78.3, 92.2, 96.8],
#     'CoCoOp': [73.4, 81.5, 89.1],
#     'CLIP-Adapter': [71.3, 73.1, 92.9],
#     'CLIP-LoRA': [83.2, 93.7, 98.0],
#     'CLIP-MoRE': [79.2, 93.5, 96.0]
# }

# # --- Plot Styling ---
# # Use a modern and clean plot style
# plt.style.use('seaborn-v0_8-whitegrid')

# # Create a figure and axes for more control over the plot
# fig, ax = plt.subplots(figsize=(10, 7))

# # Define a color palette and marker styles for better visual distinction
# colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#D55E00']
# markers = ['o', 's', '^', 'D', 'v', 'P']

# # --- Plotting the data ---
# for i, (name, accuracy) in enumerate(methods.items()):
#     ax.plot(shots, accuracy, marker=markers[i], linestyle='-', linewidth=2, markersize=8, label=name, color=colors[i])

# # --- Customizing the Plot Appearance ---
# # Set titles and labels with appropriate font sizes for clarity
# ax.set_title('Few-Shot Performance on Flowers-102', fontsize=16, fontweight='bold', pad=20)
# ax.set_xlabel('Number of Shots', fontsize=12)
# ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)

# # Set x-axis to a logarithmic scale to better visualize performance at low shots,
# # and ensure ticks appear for each shot count.
# ax.set_xscale('log')
# ax.set_xticks(shots)
# ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
# ax.tick_params(axis='both', which='major', labelsize=10)


# # Adjust y-axis to provide some space around the min/max values
# min_acc = min(min(v) for v in methods.values())
# max_acc = max(max(v) for v in methods.values())
# ax.set_ylim(min_acc - 5, max_acc + 2)


# # Place the legend outside the plot area for a cleaner look
# ax.legend(title='Method', fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left')

# # Remove top and right spines for a cleaner aesthetic
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# # Add a light grid for the y-axis
# ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)


# # --- Saving the Plot ---
# # Define the output directory and filename
# output_dir = 'graphics'
# output_filename = 'flowers_plot.png'
# output_path = os.path.join(output_dir, output_filename)

# # Create the directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)

# # Save the figure with high resolution and tight layout
# plt.savefig(output_path, dpi=300, bbox_inches='tight')

# # Optionally, display the plot
# plt.show()

# print(f"Plot successfully saved to: {output_path}")

import plotly.graph_objects as go
import plotly.io as pio
import os
from typing import Dict, List, Union

# Set a default theme for the plots
pio.templates.default = "plotly_white"

def plot_few_shot_performance(
    methods_data: Dict[str, List[Union[int, float]]],
    shots: List[int],
    dataset_name: str,
    output_dir: str = 'graphics',
    # Change the default filename to reflect the new format
    filename: str = 'performance_plot.png' 
) -> None:
    """
    Creates, displays, and saves an interactive few-shot performance plot using Plotly.

    Args:
        methods_data: A dictionary where keys are method names and values are lists of accuracies.
        shots: A list of integers representing the number of shots (x-axis).
        dataset_name: The name of the dataset for the plot title.
        output_dir: The directory to save the plot in.
        filename: The name of the output image file (e.g., 'plot.png').
    """
    print(f"Generating plot for {dataset_name}...")

    fig = go.Figure()

    # --- This part of the function (adding traces and updating layout) stays the same ---
    # ... (code for adding traces and updating layout) ...
    # Define a color palette and marker styles for better visual distinction
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['circle', 'square', 'diamond', 'cross', 'x', 'star']

    # --- Add traces for each method ---
    for i, (name, accuracy) in enumerate(methods_data.items()):
        fig.add_trace(go.Scatter(
            x=shots,
            y=accuracy,
            mode='lines+markers',
            name=name,
            line=dict(width=2.5, color=colors[i % len(colors)]),
            marker=dict(symbol=markers[i % len(markers)], size=10),
            hovertemplate=f'<b>{name}</b><br>' +
                          'Shots: %{x}<br>' +
                          'Accuracy: %{y:.2f}%<extra></extra>'
        ))

    # --- Customize the Plot Appearance ---
    min_acc = min(min(v) for v in methods_data.values())
    max_acc = max(max(v) for v in methods_data.values())

    fig.update_layout(
        title=dict(
            text=f'<b>Few-Shot Performance on {dataset_name}</b>',
            font=dict(size=20),
            x=0.5,
            pad=dict(b=20)
        ),
        xaxis=dict(
            title='Number of Shots',
            type='log',
            tickvals=shots,
            ticktext=[str(s) for s in shots],
            gridcolor='rgba(0,0,0,0.05)'
        ),
        yaxis=dict(
            title='Top-1 Accuracy (%)',
            range=[min_acc - 5, max_acc + 3], # Add padding
            gridcolor='rgba(0,0,0,0.1)'
        ),
        legend=dict(
            title='<b>Method</b>',
            yanchor="top",
            y=1.02,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255, 255, 255, 0.6)'
        ),
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=80, r=80, t=80, b=80)
    )
    # --- End of unchanged section ---

    # --- Save the Plot ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename)
    
    # --- MODIFICATION HERE: Use write_image instead of write_html ---
    fig.write_image(output_path)
    
    print(f"Plot successfully saved to: {output_path}")
    fig.show()

def plot_model_comparison_bar(
    methods_data: Dict[str, List[Union[int, float]]],
    datasets: List[str],
    output_dir: str = 'graphics',
    filename: str = 'nlp_comparison.png'
) -> None:
    """
    Creates, displays, and saves an interactive bar chart comparing model performance.

    Args:
        methods_data: A dictionary where keys are model names and values are lists of scores.
        datasets: A list of strings representing the dataset names (x-axis categories).
        output_dir: The directory to save the plot in.
        filename: The name of the output HTML file.
    """
    print("Generating NLP performance comparison bar chart...")

    fig = go.Figure()

    # Define a color palette
    colors = {'LoRA': '#1f77b4', 'MoRE': '#ff7f0e'}

    # --- Add traces for each method ---
    for method_name, scores in methods_data.items():
        fig.add_trace(go.Bar(
            x=datasets,
            y=scores,
            name=method_name,
            marker_color=colors.get(method_name),
            text=scores,
            textposition='auto',
            texttemplate='%{y:.1f}',
            hovertemplate=f'<b>{method_name}</b><br>' +
                          'Dataset: %{x}<br>' +
                          'Accuracy: %{y:.1f}%<extra></extra>'
        ))

    # --- Customize the Plot Appearance ---
    fig.update_layout(
        title=dict(
            text='<b>MoRE vs. LoRA: Commonsense Reasoning Performance</b>',
            font=dict(size=20),
            x=0.5,
            pad=dict(b=20)
        ),
        xaxis=dict(
            title='Dataset',
            tickangle=-45
        ),
        yaxis=dict(
            title='Top-1 Accuracy (%)',
            gridcolor='rgba(0,0,0,0.1)'
        ),
        legend=dict(
            title='<b>Model</b>',
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.01
        ),
        barmode='group', # Group bars for each dataset side-by-side
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=80, r=40, t=80, b=100) # Adjust bottom margin for angled labels
    )
    
    # --- Save the Plot ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename)
    fig.write_image(output_path)
    print(f"Plot successfully saved to: {output_path}")
    fig.show()


def main():
    """
    Main function to define data and generate plots.
    """
    # --- Data for the first plot (Flowers-102 full comparison) ---
    flowers_methods_data = {
        'CoOp(4)': [72.7, 86.6, 96.4],
        'CoOp(16)': [78.3, 92.2, 96.8],
        'CoCoOp': [73.4, 81.5, 89.1],
        'CLIP-Adapter': [71.3, 73.1, 92.9],
        'CLIP-LoRA': [83.2, 93.7, 98.0],
        'CLIP-MoRE': [79.2, 93.5, 96.0]
    }

    nlp_methods_data = {
        'LoRA': [82.0, 91.5, 95.0],
        'MoRE': [80.0, 92.0, 96.0]
    }

    shots = [1, 4, 16]
    
    plot_few_shot_performance(
        methods_data=flowers_methods_data,
        shots=shots,
        dataset_name="Flowers-102",
    )


   # --- Data from the Commonsense Reasoning task ---
    datasets = ["BoolQ", "PIQA", "SIQA", "Hellas.", "WinoG.", "ARC-e", "ARC-c", "OBQA"]
    
    # Using the data you provided, excluding the final "Average" score for MoRE 
    # to match the 8 datasets.
    nlp_methods_data = {
        'LoRA': [68.9, 80.7, 77.4, 78.1, 78.8, 77.8, 61.3, 74.8],
        'MoRE': [67.0, 86.4, 88.4, 97.3, 95.1, 88.5, 76.6, 79.9]
    }

    # Call the plotting function with the corrected data
    plot_model_comparison_bar(
        methods_data=nlp_methods_data,
        datasets=datasets
    )
    
if __name__ == '__main__':
    main()
