# file: jax2onnx/sandbox/mnist_vit.py
import os
import re
import shutil
import zipfile
import warnings
from typing import Dict, Any


import jax.lax
import jax
import onnx

# import torchvision
from flax import nnx

# from torch.utils.data import DataLoader
# from torchvision import transforms
from jax2onnx import to_onnx, allclose
from jax2onnx.plugins.examples.nnx.vit import VisionTransformer

# from jaxamples.vit import VisionTransformer
import orbax.checkpoint as orbax

# import matplotlib


from logging_config import configure_logging

configure_logging()

# matplotlib.use("Agg")  # Use a non-interactive backend to avoid Tkinter-related issues
# import matplotlib.pyplot as plt
# import csv

warnings.filterwarnings(
    "ignore", message="Couldn't find sharding info under RestoreArgs.*"
)


# # =============================================================================
# # Data, augmentation and model utility functions
# # =============================================================================


# def get_dataset_torch_dataloaders(batch_size: int, data_dir: str = "./data"):
#     transform = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
#     )
#     train_ds = torchvision.datasets.MNIST(
#         data_dir, train=True, download=True, transform=transform
#     )
#     test_ds = torchvision.datasets.MNIST(
#         data_dir, train=False, download=True, transform=transform
#     )
#     train_dataloader = DataLoader(
#         train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
#     )
#     test_dataloader = DataLoader(
#         test_ds, batch_size=1000, shuffle=False, num_workers=0, drop_last=True
#     )
#     return train_dataloader, test_dataloader


# def jax_collate(batch):
#     images, labels = batch
#     images = jnp.array(images.numpy())
#     labels = jnp.array(labels.numpy())
#     images = jnp.transpose(images, (0, 2, 3, 1))
#     return {"image": images, "label": labels}


# def rotate_image(image: jnp.ndarray, angle: float) -> jnp.ndarray:
#     """Rotates an image."""
#     angle_rad = jnp.deg2rad(angle)
#     cos_angle = jnp.cos(angle_rad)
#     sin_angle = jnp.sin(angle_rad)
#     center_y, center_x = jnp.array(image.shape[:2]) / 2.0
#     yy, xx = jnp.meshgrid(
#         jnp.arange(image.shape[0]), jnp.arange(image.shape[1]), indexing="ij"
#     )
#     yy = yy - center_y
#     xx = xx - center_x
#     rotated_x = cos_angle * xx + sin_angle * yy + center_x
#     rotated_y = -sin_angle * xx + cos_angle * yy + center_y
#     rotated_image = map_coordinates(
#         image[..., 0], [rotated_y.ravel(), rotated_x.ravel()], order=1, mode="constant"
#     )
#     rotated_image = rotated_image.reshape(image.shape[:2])
#     return jnp.expand_dims(rotated_image, axis=-1)


# def jax_gaussian_filter(x: jnp.ndarray, sigma: float, radius: int) -> jnp.ndarray:
#     """Performs 2D Gaussian filtering on array x using separable convolutions."""
#     size = 2 * radius + 1
#     ax = jnp.arange(-radius, radius + 1, dtype=x.dtype)
#     kernel = jnp.exp(-0.5 * (ax / sigma) ** 2)
#     kernel = kernel / jnp.sum(kernel)
#     kernel_h = kernel.reshape((size, 1, 1, 1))
#     kernel_v = kernel.reshape((1, size, 1, 1))
#     x_exp = x[None, ..., None]
#     x_filtered = jax.lax.conv_general_dilated(
#         x_exp,
#         kernel_h,
#         window_strides=(1, 1),
#         padding="SAME",
#         dimension_numbers=("NHWC", "HWIO", "NHWC"),
#     )
#     x_filtered = jax.lax.conv_general_dilated(
#         x_filtered,
#         kernel_v,
#         window_strides=(1, 1),
#         padding="SAME",
#         dimension_numbers=("NHWC", "HWIO", "NHWC"),
#     )
#     return jnp.squeeze(x_filtered, axis=(0, 3))


# @functools.partial(jax.jit, static_argnames=("radius",))
# def elastic_deform(
#     image: jnp.ndarray,
#     alpha: float,
#     sigma: float,
#     rng_key: jnp.ndarray,
#     x_grid: jnp.ndarray,
#     y_grid: jnp.ndarray,
#     radius: int,
# ) -> jnp.ndarray:
#     shape = image.shape[:2]
#     key_dx, key_dy = random.split(rng_key, 2)
#     dx = random.normal(key_dx, shape) * alpha
#     dy = random.normal(key_dy, shape) * alpha

#     dx = jax_gaussian_filter(dx, sigma, radius)
#     dy = jax_gaussian_filter(dy, sigma, radius)

#     indices = (jnp.reshape(y_grid + dy, (-1, 1)), jnp.reshape(x_grid + dx, (-1, 1)))
#     deformed_image = map_coordinates(image[..., 0], indices, order=1, mode="reflect")
#     return jnp.expand_dims(deformed_image.reshape(shape), axis=-1)


# def visualize_augmented_images(
#     ds: Dict[str, jnp.ndarray], epoch: int, num_images: int = 9
# ) -> None:
#     """
#     Displays a grid of augmented images.

#     Args:
#         ds (Dict[str, jnp.ndarray]): The dataset containing the augmented images.
#         num_images (int, optional): The number of images to display. Defaults to 9.
#     """

#     fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
#     for i, ax in enumerate(axes):
#         ax.imshow(ds["image"][i, ..., 0], cmap="gray")
#         ax.axis("off")

#     plt.savefig(f"output/augmented_images_epoch{epoch}.png")
#     plt.close(fig)


# AugmentationParams = namedtuple(
#     "AugmentationParams",
#     [
#         "max_translation",
#         "scale_min_x",
#         "scale_max_x",
#         "scale_min_y",
#         "scale_max_y",
#         "max_rotation",
#         "elastic_alpha",
#         "elastic_sigma",
#         "enable_elastic",
#         "enable_rotation",
#         "enable_scaling",
#         "enable_translation",
#     ],
# )


# @functools.partial(jax.jit, static_argnames=("augmentation_params",))
# def augment_data_batch(
#     batch: Dict[str, jnp.ndarray],
#     rng_key: jnp.ndarray,
#     augmentation_params: AugmentationParams,
# ) -> Dict[str, jnp.ndarray]:
#     images = batch["image"]
#     batch_size, height, width, channels = images.shape

#     x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
#     x_grid = jnp.array(x_grid)
#     y_grid = jnp.array(y_grid)

#     alpha_val = augmentation_params.elastic_alpha
#     sigma_val = augmentation_params.elastic_sigma
#     radius_val = math.ceil(3.0 * sigma_val)

#     def augment_single_image(image, key):
#         key1, key2, key3, key4, key5, key6 = random.split(key, 6)
#         max_translation = augmentation_params.max_translation
#         tx = random.uniform(key1, minval=-max_translation, maxval=max_translation)
#         ty = random.uniform(key2, minval=-max_translation, maxval=max_translation)
#         translation = jnp.array([ty, tx])
#         scale_factor_x = random.uniform(
#             key3,
#             minval=augmentation_params.scale_min_x,
#             maxval=augmentation_params.scale_max_x,
#         )
#         scale_factor_y = random.uniform(
#             key4,
#             minval=augmentation_params.scale_min_y,
#             maxval=augmentation_params.scale_max_y,
#         )
#         scale = jnp.array([scale_factor_y, scale_factor_x])
#         max_rotation = augmentation_params.max_rotation * (jnp.pi / 180.0)
#         rotation_angle = random.uniform(key5, minval=-max_rotation, maxval=max_rotation)

#         if augmentation_params.enable_elastic:
#             image = elastic_deform(
#                 image, alpha_val, sigma_val, key6, x_grid, y_grid, radius_val
#             )

#         if augmentation_params.enable_rotation:
#             image = rotate_image(image, jnp.rad2deg(rotation_angle))

#         if augmentation_params.enable_scaling or augmentation_params.enable_translation:
#             image = scale_and_translate(
#                 image=image,
#                 shape=(height, width, channels),
#                 spatial_dims=(0, 1),
#                 scale=(
#                     scale
#                     if augmentation_params.enable_scaling
#                     else jnp.array([1.0, 1.0])
#                 ),
#                 translation=(
#                     translation
#                     if augmentation_params.enable_translation
#                     else jnp.array([0.0, 0.0])
#                 ),
#                 method="linear",
#                 antialias=True,
#             )
#         return image

#     rng_keys = random.split(rng_key, num=batch_size)
#     augmented_images = jax.vmap(augment_single_image)(images, rng_keys)
#     return {"image": augmented_images, "label": batch["label"]}


# def create_sinusoidal_embeddings(num_patches: int, num_hiddens: int) -> jnp.ndarray:
#     position = jnp.arange(num_patches + 1)[:, jnp.newaxis]
#     div_term = jnp.exp(
#         jnp.arange(0, num_hiddens, 2) * -(jnp.log(10000.0) / num_hiddens)
#     )
#     pos_embedding = jnp.zeros((num_patches + 1, num_hiddens))
#     pos_embedding = pos_embedding.at[:, 0::2].set(jnp.sin(position * div_term))
#     pos_embedding = pos_embedding.at[:, 1::2].set(jnp.cos(position * div_term))
#     return pos_embedding[jnp.newaxis, :, :]


# def loss_fn(
#     model: nnx.Module, batch: Dict[str, jnp.ndarray], deterministic: bool = False
# ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     logits = model(batch["image"], deterministic=deterministic)
#     loss = optax.softmax_cross_entropy_with_integer_labels(
#         logits=logits, labels=batch["label"]
#     ).mean()
#     return loss, logits


# @nnx.jit
# def train_step(
#     model: nnx.Module,
#     optimizer: nnx.Optimizer,
#     metrics: nnx.MultiMetric,
#     batch: Dict[str, jnp.ndarray],
#     learning_rate: float,
#     weight_decay: float,
# ):
#     model.train()
#     grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
#     (loss, logits), grads = grad_fn(model, batch)
#     metrics.update(loss=loss, logits=logits, labels=batch["label"])
#     optimizer.update(grads, learning_rate=learning_rate, weight_decay=weight_decay)


# @nnx.jit
# def eval_step(
#     model: nnx.Module, metrics: nnx.MultiMetric, batch: Dict[str, jnp.ndarray]
# ):
#     model.eval()
#     loss, logits = loss_fn(model, batch, deterministic=True)
#     metrics.update(loss=loss, logits=logits, labels=batch["label"])


# @nnx.jit
# def pred_step(model: nnx.Module, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
#     model.eval()
#     logits = model(batch["image"], deterministic=True)
#     return jnp.argmax(logits, axis=1)


# def visualize_incorrect_classifications(
#     model: nnx.Module,
#     test_dataloader: DataLoader,
#     epoch: int,
#     figsize: Tuple[int, int] = (15, 5),
# ) -> None:
#     """
#     Displays a grid of incorrectly classified images.

#     If there are more than 50 misclassified images, the visualization is skipped.

#     Args:
#         model: The trained model.
#         test_dataloader: DataLoader for the test dataset.
#         epoch: Current epoch (used for the output filename).
#         figsize: Figure size for matplotlib.
#     """
#     incorrect_images = []
#     incorrect_labels = []
#     incorrect_preds = []

#     for batch in test_dataloader:
#         batch = jax_collate(batch)
#         preds = pred_step(model, batch)
#         incorrect_mask = preds != batch["label"]
#         if incorrect_mask.any():
#             incorrect_images.append(batch["image"][incorrect_mask])
#             incorrect_labels.append(batch["label"][incorrect_mask])
#             incorrect_preds.append(preds[incorrect_mask])

#     # Concatenate all incorrect data
#     if incorrect_images:
#         incorrect_images = jnp.concatenate(incorrect_images, axis=0)
#         incorrect_images_typed: Array = incorrect_images  # to make mypy happy
#         incorrect_labels = jnp.concatenate(incorrect_labels, axis=0)
#         incorrect_preds = jnp.concatenate(incorrect_preds, axis=0)
#     else:
#         print("No incorrect classifications found.")
#         return

#     num_images = len(incorrect_images)

#     if num_images > 50:
#         print(
#             f"Too many incorrect classifications ({num_images}). Skipping visualization."
#         )
#         return

#     # If 50 or fewer, display all
#     fig, axes = (
#         plt.subplots(1, num_images, figsize=(15, 5))
#         if num_images > 1
#         else (plt.subplots(1, num_images, figsize=figsize)[1],)
#     )
#     if num_images == 1:
#         axes = [axes]  # Ensure axes is iterable for a single subplot
#     for i, ax in enumerate(axes):
#         ax.imshow(incorrect_images_typed[i, ..., 0], cmap="gray")
#         ax.set_title(f"{incorrect_labels[i]}\nbut\n{incorrect_preds[i]}")
#         ax.axis("off")

#     plt.savefig(f"output/incorrect_classifications_epoch{epoch}.png")
#     plt.close(fig)


# # =============================================================================
# # Learning rate scheduling and optimizer creation
# # =============================================================================


# def lr_schedule(epoch: int, config: Dict) -> float:
#     total_epochs = (
#         config["training"]["start_epoch"]
#         + config["training"]["num_epochs_to_train_now"]
#     )
#     base_lr = config["training"]["base_learning_rate"]
#     # Cosine schedule computed over the full training duration
#     return 0.5 * base_lr * (1 + jnp.cos(jnp.pi * epoch / total_epochs))


# def create_optimizer(
#     model: nnx.Module, learning_rate: float, weight_decay: float
# ) -> nnx.Optimizer:
#     return nnx.Optimizer(
#         model, optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
#     )


# # =============================================================================
# # Training, evaluation, checkpointing and visualization functions
# # =============================================================================


# def compute_mean_and_spread(values: List[float]) -> Tuple[float, float]:
#     """Compute the mean and spread (standard deviation) of a list of values."""
#     mean = np.mean(values)
#     spread = np.std(values)
#     return mean, spread


# def save_test_accuracy_metrics(
#     metrics_history: Dict[str, List[float]], epoch: int
# ) -> None:
#     """Saves test accuracy, mean, and spread metrics to a CSV file."""
#     output_csv = "output/test_accuracy_metrics.csv"
#     file_exists = os.path.isfile(output_csv)
#     with open(output_csv, mode="a", newline="") as csv_file:
#         writer = csv.writer(csv_file)
#         if not file_exists:
#             writer.writerow(
#                 ["epoch", "test_accuracy", "mean_accuracy", "spread_accuracy"]
#             )
#         writer.writerow(
#             [
#                 epoch,
#                 metrics_history["test_accuracy"][-1],
#                 (
#                     metrics_history["test_accuracy_mean"][-1]
#                     if metrics_history["test_accuracy_mean"][-1] is not None
#                     else "N/A"
#                 ),
#                 (
#                     metrics_history["test_accuracy_spread"][-1]
#                     if metrics_history["test_accuracy_spread"][-1] is not None
#                     else "N/A"
#                 ),
#             ]
#         )
#     print(f"Test accuracy metrics for epoch {epoch} saved to {output_csv}")


# def load_and_plot_test_accuracy_metrics(csv_filepath: str, output_fig: str) -> None:
#     """Loads test accuracy metrics from a CSV file and generates a plot."""
#     epochs = []
#     test_acc = []
#     mean_acc = []
#     spread_acc = []
#     with open(csv_filepath, mode="r", newline="") as csv_file:
#         reader = csv.DictReader(csv_file)
#         for row in reader:
#             if float(row["mean_accuracy"]) != 0.0:
#                 epochs.append(int(row["epoch"]))
#                 test_acc.append(float(row["test_accuracy"]))
#                 mean_acc.append(float(row["mean_accuracy"]))
#                 spread_acc.append(float(row["spread_accuracy"]))
#     plt.style.use("ggplot")  # Changed to a valid Matplotlib style "ggplot"
#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.plot(epochs, test_acc, label="Test Accuracy", marker="o", linestyle="-")
#     ax.plot(
#         epochs,
#         mean_acc,
#         label="Moving Mean (last 10 epochs)",
#         marker="s",
#         linestyle="--",
#     )
#     mean_arr = np.array(mean_acc)
#     spread_arr = np.array(spread_acc)
#     ax.fill_between(
#         epochs,
#         mean_arr - spread_arr,
#         mean_arr + spread_arr,
#         color="gray",
#         alpha=0.3,
#         label="Spread (±1 std)",
#     )
#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("Accuracy")
#     ax.set_title("Test Accuracy Metrics Over Epochs")
#     ax.legend()
#     ax.grid(True)
#     fig.tight_layout()
#     plt.savefig(output_fig, dpi=300)
#     plt.close(fig)
#     print(f"Test accuracy graph saved to {output_fig}")


# def save_and_plot_test_accuracy_metrics(
#     metrics_history: Dict[str, List[float]]
# ) -> None:
#     import csv

#     # Write CSV file
#     output_csv = "output/test_accuracy_metrics.csv"
#     num_epochs = len(metrics_history["test_accuracy"])
#     with open(output_csv, mode="w", newline="") as csv_file:
#         writer = csv.writer(csv_file)
#         writer.writerow(["epoch", "test_accuracy", "mean_accuracy", "spread_accuracy"])
#         for i in range(num_epochs):
#             writer.writerow(
#                 [
#                     i,
#                     metrics_history["test_accuracy"][i],
#                     metrics_history["test_accuracy_mean"][i],
#                     metrics_history["test_accuracy_spread"][i],
#                 ]
#             )
#     print(f"Test accuracy metrics saved to {output_csv}")

#     # Generate a professional-style plot.
#     plt.style.use("seaborn-paper")
#     epochs = list(range(num_epochs))
#     test_acc = metrics_history["test_accuracy"]
#     mean_acc = metrics_history["test_accuracy_mean"]
#     spread_acc = metrics_history["test_accuracy_spread"]

#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.plot(epochs, test_acc, label="Test Accuracy", marker="o", linestyle="-")
#     ax.plot(
#         epochs,
#         mean_acc,
#         label="Moving Mean (last 10 epochs)",
#         marker="s",
#         linestyle="--",
#     )
#     # Fill the area between (mean - spread) and (mean + spread)
#     mean_arr = np.array(mean_acc)
#     spread_arr = np.array(spread_acc)
#     ax.fill_between(
#         epochs,
#         mean_arr - spread_arr,
#         mean_arr + spread_arr,
#         color="gray",
#         alpha=0.3,
#         label="Spread (±1 std)",
#     )

#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("Accuracy")
#     ax.set_title("Test Accuracy Metrics Over Epochs")
#     ax.legend()
#     ax.grid(True)
#     fig.tight_layout()
#     output_fig = "output/test_accuracy_metrics.png"
#     plt.savefig(output_fig, dpi=300)
#     plt.close(fig)
#     print(f"Test accuracy graph saved to {output_fig}")


# def train_model(
#     model: nnx.Module,
#     start_epoch: int,
#     metrics: nnx.MultiMetric,
#     config: Dict,
#     train_dataloader: DataLoader,
#     test_dataloader: DataLoader,
#     rng_key: jnp.ndarray,
# ) -> Dict[str, list]:
#     metrics_history: Dict[str, List[float]] = {
#         "train_loss": [],
#         "train_accuracy": [],
#         "test_loss": [],
#         "test_accuracy": [],
#         "test_accuracy_mean": [],
#         "test_accuracy_spread": [],
#     }
#     optimizer = create_optimizer(model, config["training"]["base_learning_rate"], 1e-4)
#     augmentation_params = AugmentationParams(**config["training"]["augmentation"])

#     for epoch in range(
#         start_epoch, start_epoch + config["training"]["num_epochs_to_train_now"]
#     ):
#         learning_rate = lr_schedule(epoch, config)
#         print(f"Epoch: {epoch}, Learning rate: {learning_rate}")
#         weight_decay = min(1e-4, learning_rate / 10)

#         metrics.reset()
#         for batch in train_dataloader:
#             batch = jax_collate(batch)
#             _, dropout_rng = random.split(rng_key)
#             batch = augment_data_batch(batch, dropout_rng, augmentation_params)
#             train_step(model, optimizer, metrics, batch, learning_rate, weight_decay)

#         # Visualize augmented images once per epoch
#         visualize_augmented_images(batch, epoch, num_images=9)

#         for metric, value in metrics.compute().items():
#             metrics_history[f"train_{metric}"].append(value.item())
#         print(
#             f"[train] epoch: {epoch}, loss: {metrics_history['train_loss'][-1]:.4f}, "
#             f"accuracy: {metrics_history['train_accuracy'][-1]:.4f}"
#         )

#         metrics.reset()
#         for test_batch in test_dataloader:
#             test_batch = jax_collate(test_batch)
#             eval_step(model, metrics, test_batch)
#         for metric, value in metrics.compute().items():
#             metrics_history[f"test_{metric}"].append(value.item())
#         metrics.reset()
#         print(
#             f"[test] epoch: {epoch}, loss: {metrics_history['test_loss'][-1]:.4f}, "
#             f"accuracy: {metrics_history['test_accuracy'][-1]:.4f}"
#         )

#         # Compute mean and spread of the accuracy over the last 10 epochs
#         if len(metrics_history["test_accuracy"]) >= 10:
#             n = len(metrics_history["test_accuracy"])
#             n_recent = min(n, 10)
#             recent_accuracies = metrics_history["test_accuracy"][-n_recent:]
#             mean_accuracy, spread_accuracy = compute_mean_and_spread(recent_accuracies)
#             print(
#                 f"[test] last {n_recent} epochs mean accuracy: {mean_accuracy:.4f}, "
#                 f"spread: {spread_accuracy:.4f}"
#             )
#             # Store these values for later logging/plotting.
#             metrics_history["test_accuracy_mean"].append(mean_accuracy)
#             metrics_history["test_accuracy_spread"].append(spread_accuracy)
#         else:
#             metrics_history["test_accuracy_mean"].append(0.0)
#             metrics_history["test_accuracy_spread"].append(0.0)

#         # Save test accuracy metrics at the end of every epoch
#         save_test_accuracy_metrics(metrics_history, epoch)

#         visualize_incorrect_classifications(model, test_dataloader, epoch)
#         save_model(model, config["training"]["checkpoint_dir"], epoch)
#         load_and_plot_test_accuracy_metrics(
#             "output/test_accuracy_metrics.csv", "output/test_accuracy_metrics.png"
#         )
#     return metrics_history


# def visualize_results(
#     metrics_history: Dict[str, list],
#     model: nnx.Module,
#     test_dataloader: DataLoader,
#     epoch: int,
# ):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
#     ax1.set_title("Loss")
#     ax2.set_title("Accuracy")
#     for dataset in ("train", "test"):
#         ax1.plot(metrics_history[f"{dataset}_loss"], label=f"{dataset}_loss")
#         ax2.plot(metrics_history[f"{dataset}_accuracy"], label=f"{dataset}_accuracy")
#     ax1.legend()
#     ax2.legend()
#     plt.savefig(f"output/results_epoch_{epoch}.png")
#     plt.close(fig)

#     for test_batch in test_dataloader:
#         test_batch = jax_collate(test_batch)
#         break
#     preds = pred_step(model, test_batch)
#     fig, axs = plt.subplots(5, 5, figsize=(12, 12))
#     for i, ax in enumerate(axs.flatten()):
#         ax.imshow(test_batch["image"][i, ..., 0], cmap="gray")
#         ax.set_title(f"Prediction: {preds[i]}, Label: {test_batch['label'][i]}")
#         ax.axis("off")
#     plt.savefig(f"output/prediction_example_epoch_{epoch}.png")
#     plt.close(fig)


# def save_model_visualization(model: nnx.Module) -> None:
#     html_content = treescope.render_to_html(model)
#     output_file = "treescope_output.html"
#     with open(output_file, "w") as file:
#         file.write(html_content)
#     print(f"TreeScope HTML saved to '{output_file}'.")


def save_model(model: nnx.Module, ckpt_dir: str, epoch: int):
    state_dir = f"{ckpt_dir}/epoch_{epoch}"
    if not os.path.exists(state_dir):
        os.makedirs(state_dir)
    keys, state = nnx.state(model, nnx.RngKey, ...)
    keys = jax.tree.map(jax.random.key_data, keys)
    checkpointer = orbax.PyTreeCheckpointer()
    checkpointer.save(state_dir, state, force=True)
    zip_path = f"{ckpt_dir}/epoch_{epoch}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(state_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=state_dir)
                zipf.write(file_path, arcname)
    shutil.rmtree(state_dir)
    print(f"Model checkpoint for epoch {epoch} saved to {zip_path}")


def load_model(model: nnx.Module, ckpt_dir: str, epoch: int) -> nnx.Module:
    zip_path = f"{ckpt_dir}/epoch_{epoch}.zip"
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Checkpoint file not found at {zip_path}")
    extract_dir = f"{ckpt_dir}/epoch_{epoch}_temp"
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    with zipfile.ZipFile(zip_path, "r") as zipf:
        zipf.extractall(extract_dir)
    keys, state = nnx.state(model, nnx.RngKey, ...)
    checkpointer = orbax.PyTreeCheckpointer()
    restored_state = checkpointer.restore(extract_dir, item=state)
    nnx.update(model, keys, restored_state)
    shutil.rmtree(extract_dir)
    print(f"Model checkpoint for epoch {epoch} loaded from {zip_path}")
    return model


def get_latest_checkpoint_epoch(ckpt_dir: str) -> int:
    ckpt_dir = os.path.abspath(ckpt_dir)
    if not os.path.exists(ckpt_dir):
        return 0
    files_and_dirs = os.listdir(ckpt_dir)
    epoch_pattern = re.compile(r"epoch_(\d+)")
    epochs = [
        int(match.group(1))
        for name in files_and_dirs
        if (match := epoch_pattern.search(name))
    ]
    return max(epochs, default=0)


# def test_onnx_model(onnx_model_path: str, test_dataloader: DataLoader) -> None:
#     """Test the exported ONNX model with ONNX Runtime and the MNIST test set."""
#     session = ort.InferenceSession(onnx_model_path)
#     input_name = session.get_inputs()[0].name
#     output_name = session.get_outputs()[0].name

#     correct = 0
#     total = 0

#     for batch in test_dataloader:
#         images, labels = batch
#         images = images.numpy()
#         labels = labels.numpy()
#         images = images.transpose(0, 2, 3, 1)  # Convert to NHWC format

#         inputs = {
#             input_name: images,
#             "deterministic": np.array(True)
#         }
#         preds = session.run([output_name], inputs)[0]
#         preds = np.argmax(preds, axis=1)

#         correct += (preds == labels).sum()
#         total += labels.shape[0]

#     accuracy = correct / total
#     print(f"ONNX model test accuracy: {accuracy:.4f}")


# =============================================================================
# Main function
# =============================================================================


def main() -> None:
    os.makedirs("output", exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    # load_and_plot_test_accuracy_metrics("output/test_accuracy_metrics.csv", "output/test_accuracy_metrics.png")

    # jax.config.update("jax_log_compiles", True)  # Keep commented out unless debugging

    # Define all configuration parameters in a hierarchical dictionary.
    # "seed" is now at the top level.
    config: Dict[str, Any] = {
        "seed": 0,
        "training": {
            "enable_training": False,
            "batch_size": 2,
            "base_learning_rate": 0.0001,
            "num_epochs_to_train_now": 2,
            "warmup_epochs": 5,
            "checkpoint_dir": os.path.abspath("./data/checkpoints/"),
            "data_dir": "./data",
            "augmentation": {
                # translation in pixels
                "enable_translation": True,
                "max_translation": 3.0,
                # scaling factors in x (horizontal) and y (vertical) directions
                "enable_scaling": True,
                "scale_min_x": 0.5,
                "scale_max_x": 1.15,
                "scale_min_y": 0.85,
                "scale_max_y": 1.15,
                # rotation in degrees
                "enable_rotation": True,
                "max_rotation": 15.0,
                # elastic local deformations
                "enable_elastic": True,
                "elastic_alpha": 0.5,  # distortion intensity
                "elastic_sigma": 0.6,  # smoothing
            },
        },
        "model": {
            "height": 28,
            "width": 28,
            "num_hiddens": 256,
            "num_layers": 6,
            "num_heads": 8,
            "mlp_dim": 512,
            "num_classes": 10,
            # "embed_dims": [32, 128, 256],
            # "kernel_size": 3,
            # "strides": [1, 2, 2],
            "embedding_type": "conv",  # "patch" or "conv"
            # "embedding_dropout_rate": 0.1,
            # "attention_dropout_rate": 0.3,
            # "mlp_dropout_rate": 0.5,
        },
        "onnx": {
            "model_name": "mnist_vit_model",
            "output_path": "docs/mnist_vit_model.onnx",
            "input_shapes": [(2, 28, 28, 1)],
            "input_params": {
                "deterministic": True,
            },
        },
    }

    # Set up the model's RNG using the top-level seed.
    config["model"]["rngs"] = nnx.Rngs(config["seed"])

    # train_dataloader, test_dataloader = get_dataset_torch_dataloaders(
    #     config["training"]["batch_size"], config["training"]["data_dir"]
    # )

    rngs = config["model"]["rngs"]
    rng_key = rngs.as_jax_rng()

    start_epoch = get_latest_checkpoint_epoch(config["training"]["checkpoint_dir"])
    config["training"]["start_epoch"] = start_epoch
    print(f"Resuming from epoch: {start_epoch}.")

    # Create the model using parameters from the config.
    model = VisionTransformer(**config["model"])

    if start_epoch > 0:
        try:
            model = load_model(model, config["training"]["checkpoint_dir"], start_epoch)
            print(f"Loaded model from epoch {start_epoch}")
        except FileNotFoundError:
            print(
                f"Checkpoint for epoch {start_epoch} not found, starting from scratch."
            )
            start_epoch = 0
            model = VisionTransformer(**config["model"])

    # metrics = nnx.MultiMetric(
    #     accuracy=nnx.metrics.Accuracy(),
    #     loss=nnx.metrics.Average("loss"),
    # )

    # if config["training"]["enable_training"]:
    #     metrics_history = train_model(
    #         model,
    #         start_epoch,
    #         metrics,
    #         config,
    #         train_dataloader,
    #         test_dataloader,
    #         rng_key,
    #     )
    #     visualize_results(
    #         metrics_history,
    #         model,
    #         test_dataloader,
    #         start_epoch + config["training"]["num_epochs_to_train_now"] - 1,
    #     )

    # onnx export
    inputs = config["onnx"]["input_shapes"]
    input_params = {"deterministic": True}
    output_path = config["onnx"]["output_path"]
    print("Exporting model to ONNX...")
    onnx_model = to_onnx(model, inputs, input_params)
    onnx.save_model(onnx_model, output_path)
    print(f"Model exported to {output_path }")

    # Test the exported ONNX model
    xs = [jax.random.normal(rng_key, tuple(shape)) for shape in inputs]

    # Correct allclose usage: pass model kwargs directly, not as jax_kwargs
    model.eval()
    result = allclose(model, output_path, xs, input_params)
    print(f"ONNX allclose result: {result}")


if __name__ == "__main__":
    main()
