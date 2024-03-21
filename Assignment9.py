import numpy as np

# Define the original image and the filter
image = np.array([[1, 2, 3, 4, 5, 6],
                  [7, 8, 9, 10, 11, 12],
                  [13, 14, 15, 16, 17, 18],
                  [19, 20, 21, 22, 23, 24],
                  [25, 26, 27, 28, 29, 30],
                  [31, 32, 33, 34, 35, 36]])

filter = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

# Function to perform convolution
def convolution(image, filter):
    image_height, image_width = image.shape
    filter_height, filter_width = filter.shape

    # Output array to store the result of convolution
    output_height = image_height - filter_height + 1
    output_width = image_width - filter_width + 1
    output = np.zeros((output_height, output_width))

    # Perform convolution
    for i in range(output_height):
        for j in range(output_width):
            # Extract the region of interest from the image
            roi = image[i:i+filter_height, j:j+filter_width]
            # Perform element-wise multiplication and sum
            output[i, j] = np.sum(roi * filter)

    return output

# Perform convolution
result = convolution(image, filter)

# Print the result
print("Result of convolution:")
print(result)
