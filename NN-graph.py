import matplotlib.pyplot as plt
import numpy as np

def draw_neural_net(ax, left, right, bottom, top, layer_sizes, layer_colors, line_width=0.7):
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(n_layers - 1)
    node_radius = v_spacing / 8.

    # Nodes
    for n, (layer_size, color) in enumerate(zip(layer_sizes, layer_colors)):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            circle = plt.Circle((left + n * h_spacing, layer_top - m * v_spacing),
                                node_radius, color=color, ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D(
                    [left + n * h_spacing, left + (n + 1) * h_spacing],
                    [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing],
                    c='k', linewidth=line_width
                )
                ax.add_artist(line)


fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
plt.subplots_adjust(wspace=0.4)  


ax1 = axes[0]
ax1.axis('off')
ax1.set_aspect('equal')
draw_neural_net(
    ax1, left=0.1, right=0.9, bottom=0.1, top=0.9,
    layer_sizes=[3, 3, 2],
    layer_colors=['orange', 'red', 'deepskyblue'],
    line_width=0.7
)
ax1.set_title("Simple Layers of Neurons", fontsize=18, pad=28)

ax2 = axes[1]
ax2.axis('off')
ax2.set_aspect('equal')
draw_neural_net(
    ax2, left=0.1, right=0.9, bottom=0.1, top=0.9,
    layer_sizes=[3, 5, 5, 5, 2],
    layer_colors=['orange', 'red', 'red', 'red', 'deepskyblue'],
    line_width=0.7
)
ax2.set_title("Deep Neural Networks", fontsize=18, pad=28)


import matplotlib.patches as mpatches
handles = [
    mpatches.Patch(color='orange', label='Input Layer'),
    mpatches.Patch(color='red', label='Hidden Layer'),
    mpatches.Patch(color='deepskyblue', label='Output Layer')
]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=14, frameon=False)

plt.show()