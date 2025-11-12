"""
Minimal ESMM (Entire Space Multi-task Model) example for CTR & CVR estimation.
Uses the bundled sample_esmm_data.csv to demonstrate multi-task training.
"""

from pathlib import Path

import pandas as pd
import tensorflow as tf


class ESMM(tf.keras.Model):
    """Implements shared-bottom ESMM with separate CTR/CVR towers."""

    def __init__(self, input_dim, shared_units=(64,), tower_units=(32, 16)):
        super().__init__()
        self.shared_layers = [tf.keras.layers.Dense(u, activation="relu") for u in shared_units]
        self.ctr_tower = [tf.keras.layers.Dense(u, activation="relu") for u in tower_units]
        self.cvr_tower = [tf.keras.layers.Dense(u, activation="relu") for u in tower_units]
        self.ctr_output = tf.keras.layers.Dense(1, activation="sigmoid")
        self.cvr_output = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.shared_layers:
            x = layer(x)

        ctr_x = x
        for layer in self.ctr_tower:
            ctr_x = layer(ctr_x)
        ctr_prob = self.ctr_output(ctr_x)

        cvr_x = x
        for layer in self.cvr_tower:
            cvr_x = layer(cvr_x)
        cvr_prob = self.cvr_output(cvr_x)

        ctcvr_prob = ctr_prob * cvr_prob
        return ctr_prob, cvr_prob, ctcvr_prob


def load_sample_dataset(csv_path, batch_size=4):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ("click", "conversion")]
    features = df[feature_cols].values.astype("float32")
    click = df["click"].values.astype("float32").reshape(-1, 1)
    conversion = df["conversion"].values.astype("float32").reshape(-1, 1)
    dataset = tf.data.Dataset.from_tensor_slices(
        (features, {"click": click, "conversion": conversion})
    )
    return dataset.shuffle(len(df)).batch(batch_size), feature_cols


def train_esmm(model, dataset, epochs=10):
    bce = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)

    for epoch in range(epochs):
        ctr_loss_metric = tf.keras.metrics.Mean()
        ctcvr_loss_metric = tf.keras.metrics.Mean()
        for batch_x, labels in dataset:
            click_labels = tf.cast(labels["click"], tf.float32)
            conversion_labels = tf.cast(labels["conversion"], tf.float32)
            with tf.GradientTape() as tape:
                ctr_pred, cvr_pred, ctcvr_pred = model(batch_x, training=True)
                ctr_loss = bce(click_labels, ctr_pred)
                ctcvr_loss = bce(conversion_labels, ctcvr_pred)
                loss = ctr_loss + ctcvr_loss
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            ctr_loss_metric.update_state(ctr_loss)
            ctcvr_loss_metric.update_state(ctcvr_loss)
        print(
            f"Epoch {epoch+1}: CTR loss={ctr_loss_metric.result().numpy():.4f}, "
            f"CTCVR loss={ctcvr_loss_metric.result().numpy():.4f}"
        )


def main():
    data_path = Path(__file__).resolve().parents[1] / "data" / "sample_esmm_data.csv"
    dataset, feature_cols = load_sample_dataset(data_path, batch_size=4)
    input_dim = len(feature_cols)
    model = ESMM(input_dim=input_dim, shared_units=(64,), tower_units=(32, 16))
    train_esmm(model, dataset, epochs=8)


if __name__ == "__main__":
    main()
