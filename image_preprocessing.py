import cv2
import numpy as np
import matplotlib.pyplot as plt


def resize_image(image, width, height):
    """
    Resize an image to the selected dimensions.
    """

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def normalize_image(image):
    """
    Normalize image pixel values between 0 and 1.
    """

    return image.astype("float32") / 255.0


def gamma_correction(image, gamma=1.0):
    """
    Apply gamma correction to enhance image brightness and contrast.
    """

    image_normalized = image / 255.0
    image_gamma_corrected = np.power(image_normalized, gamma)
    image_output = np.uint8(image_gamma_corrected * 255)

    return image_output


def equalize_histogram(image):
    """
    Apply histogram equalization to improve image contrast.
    """

    return cv2.equalizeHist(image)


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE contrast enhancement.
    """

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    return clahe.apply(image)


def invert_image(image):
    """
    Invert grayscale image intensity values.
    """

    return 255 - image


def sobel_filter(image):
    """
    Apply Sobel filtering for edge detection.
    """

    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=7)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=7)

    return sobel_x + sobel_y


def laplacian_filter(image):
    """
    Apply Laplacian filtering to detect intensity variations.
    """

    return cv2.Laplacian(image, cv2.CV_64F)


def crop_center(image, crop_size):
    """
    Crop the central region of an image.
    """

    center_y, center_x = image.shape[:2]
    center_y //= 2
    center_x //= 2

    half_crop = crop_size // 2

    return image[
        center_y - half_crop:center_y + half_crop,
        center_x - half_crop:center_x + half_crop
    ]


def window_image(image, window_size=700):
    """
    Extract a centered window from the image.
    """

    height, width = image.shape[:2]

    center_x = width // 2
    center_y = height // 2
    half_size = window_size // 2

    start_x = max(center_x - half_size, 0)
    end_x = min(center_x + half_size, width)

    start_y = max(center_y - half_size, 0)
    end_y = min(center_y + half_size, height)

    return image[start_y:end_y, start_x:end_x]


def plot_histogram(image, title="Image Histogram"):
    """
    Plot histogram, probability density function and cumulative distribution.
    """

    min_value = image.min()
    max_value = image.max()

    hist, hist_bins = np.histogram(
        image,
        bins=max_value - min_value + 1,
        range=(min_value, max_value + 1)
    )

    pdf, pdf_bins = np.histogram(
        image,
        bins=max_value - min_value + 1,
        range=(min_value, max_value + 1),
        density=True
    )

    cumulative = np.cumsum(pdf)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(title)
    axes[0].axis("off")

    axes[1].bar(np.arange(len(hist)), hist)
    axes[1].set_title("Histogram")
    axes[1].set_xlabel("Gray Intensity")
    axes[1].set_ylabel("Frequency")

    axes[2].bar(np.arange(len(pdf)), pdf)
    axes[2].set_title("Probability Density Function")
    axes[2].set_xlabel("Gray Intensity")
    axes[2].set_ylabel("Probability")

    axes[3].plot(cumulative)
    axes[3].set_title("Cumulative Distribution Function")
    axes[3].set_xlabel("Gray Intensity")
    axes[3].set_ylabel("Cumulative Frequency")

    plt.tight_layout()
    plt.show()

    return hist, pdf, cumulative
