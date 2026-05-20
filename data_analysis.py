import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import cv2
import os


def create_cardiomegaly_labels(csv_path, output_path):
    """
    Create a binary label for cardiomegaly based on the original dataset labels.
    """

    df = pd.read_csv(csv_path, sep=";")

    df_filtered = df[["Image Index", "Finding Labels"]].copy()

    df_filtered["has_cardiomegaly"] = df_filtered["Finding Labels"].apply(
        lambda row: int("cardiomegaly" in str(row).lower())
    )

    df_filtered.to_csv(output_path, index=False)

    return df_filtered


def plot_class_distribution(labels_csv):
    """
    Plot the distribution of cardiomegaly and non-cardiomegaly images.
    """

    df = pd.read_csv(labels_csv)

    cardiomegaly_count = df[df["has_cardiomegaly"] == 1].shape[0]
    non_cardiomegaly_count = df[df["has_cardiomegaly"] == 0].shape[0]

    labels = ["Cardiomegaly", "No Cardiomegaly"]
    sizes = [cardiomegaly_count, non_cardiomegaly_count]
    explode = (0.1, 0)

    plt.figure(figsize=(6, 6))
    plt.pie(
        sizes,
        explode=explode,
        labels=labels,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90
    )
    plt.axis("equal")
    plt.title("Cardiomegaly Distribution in the Dataset")
    plt.show()


def plot_cardiomegaly_by_sex(metadata_csv):
    """
    Analyze cardiomegaly distribution by patient sex.
    """

    df = pd.read_csv(metadata_csv)

    cardiomegaly_df = df[df["Finding Labels"] == "Cardiomegaly"]

    group = cardiomegaly_df["Patient Sex"].value_counts()
    group = group.to_frame(name="Cardiomegaly")

    group.plot(
        kind="bar",
        figsize=(8, 5),
        legend=False,
        title="Cardiomegaly by Patient Sex"
    )

    plt.xlabel("Patient Sex")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.show()


def analyze_image_dimensions(image_folder):
    """
    Count and plot the distribution of image dimensions in the dataset.
    """

    dimensions_counter = Counter()
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)

            if image is not None:
                height, width = image.shape[:2]
                dimensions_counter[(width, height)] += 1

    sorted_dimensions = sorted(
        dimensions_counter.items(),
        key=lambda x: x[1],
        reverse=True
    )

    labels = [f"{w}x{h}" for (w, h), _ in sorted_dimensions]
    values = [count for _, count in sorted_dimensions]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Image Dimensions")
    plt.ylabel("Number of Images")
    plt.title("Distribution of Image Dimensions")
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.show()

    return dimensions_counter
