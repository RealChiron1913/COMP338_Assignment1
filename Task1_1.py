import numpy as np
import time


def convolution_2d(image, kernel):
    # Get image dimensions
    (iH, iW) = image.shape[:2]
    # Get kernel dimensions
    (kH, kW) = kernel.shape[:2]

    # Calculate padding values to keep the output image dimensions same as input
    pad = (kW - 1) // 2
    # Pad the image
    if len(image.shape) == 3:  # RGB image
        image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)),
                       'constant')
    else:  # Grayscale image
        image = np.pad(image, ((pad, pad), (pad, pad)),
                       'edge')

    # Initialize the output image
    output = np.zeros((iH, iW), dtype="float32")
    if len(image.shape) == 3:  # RGB image
        output = np.zeros((iH, iW, 3), dtype="float32")

    start_time = time.time()
    # Apply convolution using nested loops
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # Extract the region of interest from the image
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # Apply the convolution operation
            if len(roi.shape) == 3:  # RGB image
                for channel in range(roi.shape[2]):
                    output[y - pad, x - pad, channel] \
                        = np.sum(roi[:, :, channel] * kernel)
            else:  # Grayscale image
                output[y - pad, x - pad] = np.sum(roi * kernel)
    end_time = time.time()
    print("Time taken for loop: ", end_time - start_time)

    output = np.clip(output, 0, 255).astype("uint8")

    return output


