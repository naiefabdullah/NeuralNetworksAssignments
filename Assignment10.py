import numpy as np
from scipy.signal import convolve2d
from skimage import io, color
from skimage.exposure import rescale_intensity

def depthwise_convolution(image, kernel):
    """Apply depthwise convolution to a grayscale image using the provided kernel."""
    convoluted_image = convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)
    return convoluted_image

def pointwise_convolution(image, kernel):
    """Apply pointwise convolution to an image using the provided 1x1 kernel."""
    if kernel.shape != (1, 1, image.shape[-1]):
        raise ValueError("Kernel must be of shape (1, 1, num_channels).")
    
    # Apply pointwise convolution by multiplying each channel with the corresponding kernel weight
    convoluted_image = np.empty_like(image)
    for i in range(image.shape[-1]):
        convoluted_image[:, :, i] = image[:, :, i] * kernel[0, 0, i]
    return convoluted_image

def apply_convolution(image_path, kernel, convolution_type):
    """Load an image, apply the chosen convolution, rescale intensity, save and return the output path."""
    # Load the image
    image = io.imread(image_path)
    original_dtype = image.dtype

    # Check the convolution type
    if convolution_type == "depthwise":
        # Convert to grayscale if it's not already
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = color.rgb2gray(image)
        # Apply depthwise convolution
        convoluted_image = depthwise_convolution(image, kernel)
    elif convolution_type == "pointwise":
        if len(image.shape) != 3 or kernel.shape != (1, 1, image.shape[-1]):
            raise ValueError("Pointwise convolution requires a 3-channel image and a compatible kernel.")
        # Apply pointwise convolution
        convoluted_image = pointwise_convolution(image, kernel)
    else:
        raise ValueError("Invalid convolution type specified.")

    # Rescale the intensity of the image to span the full uint8 range [0, 255]
    convoluted_image_rescaled = rescale_intensity(convoluted_image, in_range='image', out_range=(0, 255))
    convoluted_image_rescaled = convoluted_image_rescaled.astype(original_dtype)

    # Save the convoluted image
    output_path = image_path.replace(".jpg", f"_{convolution_type}_convolution.png")
    io.imsave(output_path, convoluted_image_rescaled)

    return output_path

# Define the path to your image and the kernels for each convolution type
image_path = "/Users/naiefabdullahshakil/Desktop/mishima-un-grayscale-photography-wallpaper.jpg"
depthwise_kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)  # Example depthwise kernel
pointwise_kernel = np.array([[[0.5]], [[1.0]], [[0.5]]], dtype=np.float32)  # Example pointwise kernel

# Apply depthwise convolution
output_depthwise = apply_convolution(image_path, depthwise_kernel, "depthwise")
print(f"Depthwise convoluted image saved to: {output_depthwise}")

# Apply pointwise convolution
# output_pointwise = apply_convolution(image_path, pointwise_kernel, "pointwise")
# print(f"Pointwise convoluted image saved to: {output_pointwise}")
