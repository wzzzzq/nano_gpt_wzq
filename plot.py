#!/usr/bin/env python3
# filepath: /home/sichongjie_sub3/nano_gpt_wzq/plot_loss.py

import re
import os
import matplotlib.pyplot as plt
import numpy as np

# File paths
log_file = '/home/sichongjie_sub3/nano_gpt_wzq/out/training_gpt2-124M.log'
output_file = '/home/sichongjie_sub3/nano_gpt_wzq/out/loss_plot.png'

# Lists to store data
steps = []
train_losses = []
val_losses = []

# Parse the log file
pattern = r"step (\d+): train loss ([0-9.]+), val loss ([0-9.]+)"
with open(log_file, 'r') as f:
    for line in f:
        match = re.search(pattern, line)
        if match:
            step = int(match.group(1))
            train_loss = float(match.group(2))
            val_loss = float(match.group(3))
            
            steps.append(step)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

# Create plot
plt.figure(figsize=(12, 6))
plt.plot(steps, train_losses, 'b-', label='Training Loss', marker='o', markersize=4)
plt.plot(steps, val_losses, 'r--', label='Validation Loss', marker='x', markersize=4)

# Add labels and title
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Training and Validation Loss with Flash Attention')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Mark minimum points
min_train_idx = np.argmin(train_losses)
min_val_idx = np.argmin(val_losses)

plt.annotate(f'Min Train: {train_losses[min_train_idx]:.4f} (step {steps[min_train_idx]})',
             xy=(steps[min_train_idx], train_losses[min_train_idx]),
             xytext=(steps[min_train_idx]+200, train_losses[min_train_idx]+0.3),
             arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5),
             fontsize=9)

plt.annotate(f'Min Val: {val_losses[min_val_idx]:.4f} (step {steps[min_val_idx]})',
             xy=(steps[min_val_idx], val_losses[min_val_idx]),
             xytext=(steps[min_val_idx]-500, val_losses[min_val_idx]+0.4),
             arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
             fontsize=9)

# Save figure
plt.tight_layout()
plt.savefig(output_file, dpi=300)
print(f"Plot saved to {output_file}")