"""
TensorFlow GAN Project
======================
A complete Generative Adversarial Network implementation using TensorFlow.

Features:
- DCGAN architecture
- TensorFlow 2.x
- CIFAR10 dataset loader
- Generator model
- Discriminator model
- Training loop
- Checkpoint saving
- Image saving
- Metrics tracking
- Config system
- Utility functions
- Visualization tools
- Command line training interface

Run:
    python gan_project.py

Requirements:
    pip install tensorflow matplotlib numpy pillow
"""

import os
import time
import math
import json
import random
import pathlib
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Main configuration class."""

    BUFFER_SIZE = 50000
    BATCH_SIZE = 128
    IMAGE_SIZE = 32
    CHANNELS = 3
    LATENT_DIM = 100

    EPOCHS = 50
    LEARNING_RATE = 0.0002
    BETA_1 = 0.5

    SAVE_INTERVAL = 5
    SAMPLE_INTERVAL = 1

    OUTPUT_DIR = "gan_output"
    CHECKPOINT_DIR = "gan_output/checkpoints"
    IMAGE_DIR = "gan_output/generated_images"
    LOG_DIR = "gan_output/logs"

    USE_LABEL_SMOOTHING = True
    LABEL_NOISE = 0.05

    SEED = 42


config = Config()


# ============================================================
# RANDOM SEED
# ============================================================

np.random.seed(config.SEED)
random.seed(config.SEED)
tf.random.set_seed(config.SEED)


# ============================================================
# DIRECTORY MANAGEMENT
# ============================================================


def create_directories():
    """Create output directories."""

    directories = [
        config.OUTPUT_DIR,
        config.CHECKPOINT_DIR,
        config.IMAGE_DIR,
        config.LOG_DIR,
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


create_directories()


# ============================================================
# DATA LOADING
# ============================================================


def normalize_images(images):
    """Normalize image pixels to [-1, 1]."""

    images = images.astype("float32")
    images = (images - 127.5) / 127.5

    return images



def load_dataset():
    """Load CIFAR10 dataset."""

    (train_images, train_labels), (_, _) = cifar10.load_data()

    train_images = normalize_images(train_images)

    dataset = tf.data.Dataset.from_tensor_slices(train_images)
    dataset = dataset.shuffle(config.BUFFER_SIZE)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ============================================================
# IMAGE UTILITIES
# ============================================================


def denormalize_image(image):
    """Convert image back to [0,255]."""

    image = (image + 1.0) * 127.5
    image = np.clip(image, 0, 255)

    return image.astype(np.uint8)



def save_single_image(image_array, path):
    """Save a single image."""

    image_array = denormalize_image(image_array)

    image = Image.fromarray(image_array)
    image.save(path)



def create_image_grid(images, grid_size=4):
    """Create image grid."""

    images = denormalize_image(images)

    h, w = config.IMAGE_SIZE, config.IMAGE_SIZE

    canvas = np.zeros(
        (grid_size * h, grid_size * w, config.CHANNELS),
        dtype=np.uint8,
    )

    index = 0

    for y in range(grid_size):
        for x in range(grid_size):
            if index >= len(images):
                break

            canvas[
                y * h:(y + 1) * h,
                x * w:(x + 1) * w,
            ] = images[index]

            index += 1

    return canvas



def save_image_grid(images, epoch):
    """Save generated image grid."""

    grid = create_image_grid(images)

    path = os.path.join(
        config.IMAGE_DIR,
        f"epoch_{epoch:04d}.png"
    )

    Image.fromarray(grid).save(path)


# ============================================================
# GENERATOR MODEL
# ============================================================


def build_generator():
    """Build generator network."""

    model = tf.keras.Sequential(name="generator")

    model.add(layers.Input(shape=(config.LATENT_DIM,)))

    model.add(layers.Dense(4 * 4 * 512, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((4, 4, 512)))

    model.add(
        layers.Conv2DTranspose(
            256,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            128,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            64,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2D(
            config.CHANNELS,
            kernel_size=3,
            padding="same",
            activation="tanh",
        )
    )

    return model


# ============================================================
# DISCRIMINATOR MODEL
# ============================================================


def build_discriminator():
    """Build discriminator network."""

    model = tf.keras.Sequential(name="discriminator")

    model.add(
        layers.Input(
            shape=(
                config.IMAGE_SIZE,
                config.IMAGE_SIZE,
                config.CHANNELS,
            )
        )
    )

    model.add(
        layers.Conv2D(
            64,
            kernel_size=4,
            strides=2,
            padding="same",
        )
    )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(
        layers.Conv2D(
            128,
            kernel_size=4,
            strides=2,
            padding="same",
        )
    )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(
        layers.Conv2D(
            256,
            kernel_size=4,
            strides=2,
            padding="same",
        )
    )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


# ============================================================
# LOSS FUNCTIONS
# ============================================================


cross_entropy = tf.keras.losses.BinaryCrossentropy(
    from_logits=True
)



def discriminator_loss(real_output, fake_output):
    """Discriminator loss."""

    real_labels = tf.ones_like(real_output)
    fake_labels = tf.zeros_like(fake_output)

    if config.USE_LABEL_SMOOTHING:
        real_labels *= 0.9

    real_loss = cross_entropy(real_labels, real_output)
    fake_loss = cross_entropy(fake_labels, fake_output)

    total_loss = real_loss + fake_loss

    return total_loss



def generator_loss(fake_output):
    """Generator loss."""

    labels = tf.ones_like(fake_output)

    return cross_entropy(labels, fake_output)


# ============================================================
# METRICS TRACKER
# ============================================================


class MetricTracker:
    """Track training metrics."""

    def __init__(self):
        self.generator_losses = []
        self.discriminator_losses = []
        self.epoch_times = []

    def add(self, g_loss, d_loss, epoch_time):
        self.generator_losses.append(float(g_loss))
        self.discriminator_losses.append(float(d_loss))
        self.epoch_times.append(float(epoch_time))

    def save(self, path):
        data = {
            "generator_losses": self.generator_losses,
            "discriminator_losses": self.discriminator_losses,
            "epoch_times": self.epoch_times,
        }

        with open(path, "w") as file:
            json.dump(data, file, indent=4)

    def plot(self):
        plt.figure(figsize=(10, 5))

        plt.plot(self.generator_losses, label="Generator")
        plt.plot(self.discriminator_losses, label="Discriminator")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("GAN Training Loss")
        plt.legend()

        path = os.path.join(config.OUTPUT_DIR, "loss_plot.png")

        plt.savefig(path)
        plt.close()


# ============================================================
# GAN TRAINER
# ============================================================


class GANTrainer:
    """Main GAN trainer."""

    def __init__(self):
        self.generator = build_generator()
        self.discriminator = build_discriminator()

        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE,
            beta_1=config.BETA_1,
        )

        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE,
            beta_1=config.BETA_1,
        )

        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator,
        )

        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            config.CHECKPOINT_DIR,
            max_to_keep=5,
        )

        self.metric_tracker = MetricTracker()

        self.seed = tf.random.normal([16, config.LATENT_DIM])

    @tf.function
    def train_step(self, real_images):
        """Single training step."""

        batch_size = tf.shape(real_images)[0]

        noise = tf.random.normal(
            [batch_size, config.LATENT_DIM]
        )

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(
                real_images,
                training=True,
            )

            fake_output = self.discriminator(
                generated_images,
                training=True,
            )

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss,
            self.generator.trainable_variables,
        )

        gradients_of_discriminator = disc_tape.gradient(
            disc_loss,
            self.discriminator.trainable_variables,
        )

        self.generator_optimizer.apply_gradients(
            zip(
                gradients_of_generator,
                self.generator.trainable_variables,
            )
        )

        self.discriminator_optimizer.apply_gradients(
            zip(
                gradients_of_discriminator,
                self.discriminator.trainable_variables,
            )
        )

        return gen_loss, disc_loss

    def generate_samples(self, epoch):
        """Generate image samples."""

        predictions = self.generator(
            self.seed,
            training=False,
        )

        save_image_grid(predictions.numpy(), epoch)

    def save_checkpoint(self):
        """Save checkpoint."""

        self.checkpoint_manager.save()

    def load_latest_checkpoint(self):
        """Load latest checkpoint if available."""

        latest = self.checkpoint_manager.latest_checkpoint

        if latest:
            self.checkpoint.restore(latest)
            print(f"Loaded checkpoint: {latest}")
        else:
            print("No checkpoint found.")

    def train(self, dataset):
        """Main training loop."""

        print("Starting GAN training...")

        for epoch in range(config.EPOCHS):
            start_time = time.time()

            gen_losses = []
            disc_losses = []

            for batch in dataset:
                g_loss, d_loss = self.train_step(batch)

                gen_losses.append(g_loss.numpy())
                disc_losses.append(d_loss.numpy())

            avg_g_loss = np.mean(gen_losses)
            avg_d_loss = np.mean(disc_losses)

            epoch_time = time.time() - start_time

            self.metric_tracker.add(
                avg_g_loss,
                avg_d_loss,
                epoch_time,
            )

            print(
                f"Epoch {epoch + 1}/{config.EPOCHS} | "
                f"G Loss: {avg_g_loss:.4f} | "
                f"D Loss: {avg_d_loss:.4f} | "
                f"Time: {epoch_time:.2f}s"
            )

            if (epoch + 1) % config.SAMPLE_INTERVAL == 0:
                self.generate_samples(epoch + 1)

            if (epoch + 1) % config.SAVE_INTERVAL == 0:
                self.save_checkpoint()

        self.metric_tracker.plot()

        metrics_path = os.path.join(
            config.OUTPUT_DIR,
            "metrics.json",
        )

        self.metric_tracker.save(metrics_path)

        print("Training completed.")


# ============================================================
# MODEL SUMMARY FUNCTIONS
# ============================================================


def print_model_summaries(generator, discriminator):
    """Print model summaries."""

    print("\n" + "=" * 60)
    print("GENERATOR SUMMARY")
    print("=" * 60)

    generator.summary()

    print("\n" + "=" * 60)
    print("DISCRIMINATOR SUMMARY")
    print("=" * 60)

    discriminator.summary()


# ============================================================
# LATENT SPACE UTILITIES
# ============================================================


class LatentSpaceExplorer:
    """Explore latent vectors."""

    def __init__(self, generator):
        self.generator = generator

    def random_vector(self):
        return tf.random.normal([1, config.LATENT_DIM])

    def interpolate(self, vector_a, vector_b, steps=10):
        vectors = []

        for alpha in np.linspace(0, 1, steps):
            vector = (1 - alpha) * vector_a + alpha * vector_b
            vectors.append(vector)

        return vectors

    def generate_interpolation_grid(self):
        vector_a = self.random_vector()
        vector_b = self.random_vector()

        vectors = self.interpolate(vector_a, vector_b, steps=16)

        images = []

        for vector in vectors:
            image = self.generator(vector, training=False)
            images.append(image[0].numpy())

        grid = create_image_grid(np.array(images))

        path = os.path.join(
            config.OUTPUT_DIR,
            "latent_interpolation.png",
        )

        Image.fromarray(grid).save(path)


# ============================================================
# IMAGE EVALUATION UTILITIES
# ============================================================


class ImageEvaluator:
    """Evaluate generated images."""

    def __init__(self, generator):
        self.generator = generator

    def generate_batch(self, batch_size=64):
        noise = tf.random.normal(
            [batch_size, config.LATENT_DIM]
        )

        images = self.generator(noise, training=False)

        return images.numpy()

    def compute_pixel_statistics(self, images):
        mean = np.mean(images)
        std = np.std(images)
        minimum = np.min(images)
        maximum = np.max(images)

        return {
            "mean": float(mean),
            "std": float(std),
            "min": float(minimum),
            "max": float(maximum),
        }

    def evaluate(self):
        images = self.generate_batch()

        stats = self.compute_pixel_statistics(images)

        print("\nGenerated Image Statistics")
        print("-" * 40)

        for key, value in stats.items():
            print(f"{key}: {value:.4f}")


# ============================================================
# DATA VISUALIZATION
# ============================================================


class DatasetVisualizer:
    """Visualize training dataset."""

    def __init__(self, dataset):
        self.dataset = dataset

    def save_preview(self):
        for batch in self.dataset.take(1):
            images = batch[:16].numpy()

            grid = create_image_grid(images)

            path = os.path.join(
                config.OUTPUT_DIR,
                "dataset_preview.png",
            )

            Image.fromarray(grid).save(path)

            break


# ============================================================
# TRAINING REPORT GENERATOR
# ============================================================


class TrainingReport:
    """Generate text report."""

    def __init__(self, trainer):
        self.trainer = trainer

    def create(self):
        report = []

        report.append("GAN Training Report")
        report.append("=" * 50)
        report.append("")

        report.append(f"Epochs: {config.EPOCHS}")
        report.append(f"Batch Size: {config.BATCH_SIZE}")
        report.append(f"Latent Dim: {config.LATENT_DIM}")
        report.append(f"Learning Rate: {config.LEARNING_RATE}")
        report.append("")

        if self.trainer.metric_tracker.generator_losses:
            final_g = self.trainer.metric_tracker.generator_losses[-1]
            final_d = self.trainer.metric_tracker.discriminator_losses[-1]

            report.append(f"Final Generator Loss: {final_g:.4f}")
            report.append(f"Final Discriminator Loss: {final_d:.4f}")

        report.append("")
        report.append("Generated Files:")

        for filename in os.listdir(config.OUTPUT_DIR):
            report.append(f"- {filename}")

        path = os.path.join(config.OUTPUT_DIR, "report.txt")

        with open(path, "w") as file:
            file.write("\n".join(report))


# ============================================================
# ADVANCED NOISE FUNCTIONS
# ============================================================


class NoiseGenerator:
    """Generate different noise types."""

    @staticmethod
    def gaussian(batch_size):
        return tf.random.normal(
            [batch_size, config.LATENT_DIM]
        )

    @staticmethod
    def uniform(batch_size):
        return tf.random.uniform(
            [batch_size, config.LATENT_DIM],
            minval=-1,
            maxval=1,
        )

    @staticmethod
    def truncated(batch_size):
        noise = tf.random.normal(
            [batch_size, config.LATENT_DIM]
        )

        noise = tf.clip_by_value(noise, -2, 2)

        return noise


# ============================================================
# CALLBACK SYSTEM
# ============================================================


class Callback:
    """Base callback."""

    def on_epoch_end(self, epoch, logs=None):
        pass


class PrintCallback(Callback):
    """Print callback."""

    def on_epoch_end(self, epoch, logs=None):
        print(f"Callback triggered at epoch {epoch}")


class CallbackManager:
    """Manage callbacks."""

    def __init__(self):
        self.callbacks = []

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def trigger_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)


# ============================================================
# SIMPLE INFERENCE INTERFACE
# ============================================================


class GANInference:
    """Inference helper."""

    def __init__(self, generator):
        self.generator = generator

    def generate_image(self):
        noise = tf.random.normal([1, config.LATENT_DIM])

        image = self.generator(noise, training=False)[0]

        return image.numpy()

    def save_random_image(self, filename="random.png"):
        image = self.generate_image()

        path = os.path.join(config.OUTPUT_DIR, filename)

        save_single_image(image, path)


# ============================================================
# CONFIG SAVE/LOAD
# ============================================================


class ConfigManager:
    """Save and load configuration."""

    @staticmethod
    def save_config():
        config_dict = {
            key: value
            for key, value in Config.__dict__.items()
            if not key.startswith("__")
            and not callable(value)
        }

        path = os.path.join(config.OUTPUT_DIR, "config.json")

        with open(path, "w") as file:
            json.dump(config_dict, file, indent=4)


# ============================================================
# ASCII BANNER
# ============================================================


BANNER = """
============================================================
  _______              _____
 |__   __|            / ____|
    | | ___ _ __  ___| |  __  __ _ _ __
    | |/ _ \ '_ \/ __| | |_ |/ _` | '_ \\
    | |  __/ | | \__ \ |__| | (_| | | | |
    |_|\___|_| |_|___/\_____|\__,_|_| |_|

 TensorFlow GAN Training System
============================================================
"""


# ============================================================
# MAIN FUNCTION
# ============================================================


def main():
    """Program entry point."""

    print(BANNER)

    ConfigManager.save_config()

    dataset = load_dataset()

    visualizer = DatasetVisualizer(dataset)
    visualizer.save_preview()

    trainer = GANTrainer()

    print_model_summaries(
        trainer.generator,
        trainer.discriminator,
    )

    trainer.train(dataset)

    explorer = LatentSpaceExplorer(trainer.generator)
    explorer.generate_interpolation_grid()

    evaluator = ImageEvaluator(trainer.generator)
    evaluator.evaluate()

    inference = GANInference(trainer.generator)
    inference.save_random_image()

    report = TrainingReport(trainer)
    report.create()

    print("\nAll tasks completed successfully.")


# ============================================================
# EXTRA DEMO UTILITIES
# ============================================================


class DemoUtilities:
    """Additional demo helper functions."""

    @staticmethod
    def print_separator():
        print("-" * 60)

    @staticmethod
    def countdown(seconds=3):
        for i in range(seconds, 0, -1):
            print(f"Starting in {i}...")
            time.sleep(1)

    @staticmethod
    def system_info():
        print("TensorFlow Version:", tf.__version__)
        print("GPU Available:", tf.config.list_physical_devices("GPU"))


# ============================================================
# EXTRA IMAGE OPERATIONS
# ============================================================


class ImageOperations:
    """Image processing operations."""

    @staticmethod
    def flip_horizontal(image):
        return np.fliplr(image)

    @staticmethod
    def flip_vertical(image):
        return np.flipud(image)

    @staticmethod
    def rotate(image):
        return np.rot90(image)

    @staticmethod
    def brighten(image, factor=1.1):
        image = image * factor
        image = np.clip(image, -1, 1)
        return image


# ============================================================
# MODEL EXPORTER
# ============================================================


class ModelExporter:
    """Export trained models."""

    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator

    def export_generator(self):
        path = os.path.join(
            config.OUTPUT_DIR,
            "generator_model",
        )

        self.generator.save(path)

    def export_discriminator(self):
        path = os.path.join(
            config.OUTPUT_DIR,
            "discriminator_model",
        )

        self.discriminator.save(path)


# ============================================================
# TRAINING TIMER
# ============================================================


class TrainingTimer:
    """Measure training duration."""

    def __init__(self):
        self.start = None
        self.end = None

    def start_timer(self):
        self.start = time.time()

    def stop_timer(self):
        self.end = time.time()

    def duration(self):
        if self.start is None or self.end is None:
            return 0

        return self.end - self.start


# ============================================================
# FINAL EXECUTION
# ============================================================


if __name__ == "__main__":
    DemoUtilities.print_separator()
    DemoUtilities.system_info()
    DemoUtilities.countdown(1)

    timer = TrainingTimer()

    timer.start_timer()

    main()

    timer.stop_timer()

    print(f"Total runtime: {timer.duration():.2f} seconds")

    DemoUtilities.print_separator()
