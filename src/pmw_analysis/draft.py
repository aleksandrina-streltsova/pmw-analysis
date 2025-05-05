import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load background image using PIL
img = Image.open('background.png')
img = img.convert('RGB')  # Ensure it's RGB
img = np.array(img)

# Normalize data to [0, 1] range for histogram overlay
height, width, _ = img.shape

# Generate dummy 2D data (in normalized space)
x = np.random.normal(loc=0.5, scale=0.15, size=10000)
y = np.random.normal(loc=0.5, scale=0.15, size=10000)

# Compute 2D histogram
bins = 100
hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]])

# Normalize histogram for alpha mask
alpha = hist / hist.max()
alpha = np.flipud(alpha)  # Flip for correct orientation

# Create heatmap image (red channel)
heatmap = np.zeros((bins, bins, 3))
heatmap[..., 0] = 1  # Full red

# Plot
fig, ax = plt.subplots()
ax.imshow(img, extent=[0, 1, 0, 1], aspect='auto')  # Background image
ax.imshow(heatmap, extent=[0, 1, 0, 1], alpha=alpha, aspect='auto')  # Heatmap with transparency

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.tight_layout()
plt.show()
