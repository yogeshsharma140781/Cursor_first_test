from PIL import Image, ImageChops
import numpy as np

# File names
original_img = 'layout_visualization_page_1.png'
translated_img = 'translated_layout_page_1.png'
diff_img = 'diff_page_1.png'

# Load images
img1 = Image.open(original_img).convert('RGB')
img2 = Image.open(translated_img).convert('RGB')

# Ensure images are the same size
if img1.size != img2.size:
    print(f"Image sizes do not match: {img1.size} vs {img2.size}")
    # Resize translated to match original for diffing
    img2 = img2.resize(img1.size)

# Compute absolute difference
np_img1 = np.array(img1)
np_img2 = np.array(img2)
diff = np.abs(np_img1.astype(int) - np_img2.astype(int)).astype(np.uint8)

# Highlight differences in red
highlight = np.zeros_like(np_img1)
highlight[..., 0] = diff.max(axis=2)  # Red channel = max diff
highlight_img = Image.fromarray(highlight)

# Overlay highlight on original for context
blended = Image.blend(img1, highlight_img, alpha=0.5)
blended.save(diff_img)

# Quantify the difference
red_pixels = np.sum(highlight[..., 0] > 32)  # threshold to ignore minor noise
total_pixels = highlight[..., 0].size
percent_diff = 100 * red_pixels / total_pixels
print(f"Visual diff saved as {diff_img}")
print(f"{percent_diff:.2f}% of pixels differ between original and translated layouts.")

# Warn if above threshold
diff_threshold = 10.0  # percent
if percent_diff > diff_threshold:
    print(f"WARNING: Diff exceeds threshold of {diff_threshold}%! Layout or content may not match.")
else:
    print("Diff is within acceptable limits.") 