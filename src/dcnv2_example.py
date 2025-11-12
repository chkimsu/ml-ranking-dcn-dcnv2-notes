"""
Minimal DCNv2 example that trains on the bundled sample_ranking_data.csv.
Shows how to combine full-matrix cross layers with a deep tower.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


class CrossLayerFull(tf.keras.layers.Layer):
    """Full-matrix cross layer used in DCNv2."""

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(self.input_dim, self.input_dim),
            initializer="glorot_uniform",
            name="cross_w",
        )
        self.b = self.add_weight(
            shape=(self.input_dim,),
            initializer="zeros",
            name="cross_b",
        )

    def call(self, x0, x):
        xw = tf.matmul(x, self.w)
        return x0 * (xw + self.b) + x


class DCNv2(tf.keras.Model):
    def __init__(self, input_dim, cross_layers=3, deep_units=(64, 32)):
        super().__init__()
        self.cross_layers = [CrossLayerFull(input_dim) for _ in range(cross_layers)]
        self.deep_layers = [tf.keras.layers.Dense(units, activation="relu") for units in deep_units]
        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        x0 = inputs
        x_cross = x0
        for layer in self.cross_layers:
            x_cross = layer(x0, x_cross)
        x_deep = x0
        for layer in self.deep_layers:
            x_deep = layer(x_deep)
        combined = tf.concat([x_cross, x_deep], axis=-1)
        return self.output_layer(combined)


def load_sample_dataset(csv_path, batch_size=4):
    df = pd.read_csv(csv_path)
    features = df.drop(columns=["label"]).values.astype(np.float32)
    labels = df[["label"]].values.astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    return dataset.shuffle(len(df)).batch(batch_size)


def main():
    data_path = Path(__file__).resolve().parents[1] / "data" / "sample_ranking_data.csv"
    dataset = load_sample_dataset(data_path, batch_size=4)
    input_dim = next(iter(dataset))[0].shape[-1]

    model = DCNv2(input_dim=input_dim, cross_layers=3, deep_units=(64, 32))
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)

    for epoch in range(5):
        mean_loss = tf.keras.metrics.Mean()
        for batch_x, batch_y in dataset:
            with tf.GradientTape() as tape:
                preds = model(batch_x, training=True)
                loss = loss_fn(batch_y, preds)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            mean_loss.update_state(loss)
        print(f"Epoch {epoch+1}: loss={mean_loss.result().numpy():.4f}")


if __name__ == "__main__":
    main()
