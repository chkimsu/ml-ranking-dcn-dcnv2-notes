"""
Example training loop for DCN/DCNv2 ranking model using TensorFlow.
This demonstrates building a simple DCN model and training it on a dummy dataset.
"""

import tensorflow as tf
import numpy as np

class DCN(tf.keras.Model):
    def __init__(self, input_dim, cross_layers=2, deep_units=[64, 32]):
        super().__init__()
        # Define cross network layers
        self.cross_layers = []
        for _ in range(cross_layers):
            self.cross_layers.append(tf.keras.layers.Dense(1, activation=None))
        # Define deep network layers
        self.deep_layers = [tf.keras.layers.Dense(units, activation="relu") for units in deep_units]
        # Output layer
        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        # Cross network: explicit feature interactions
        x0 = inputs
        cross_x = x0
        for dense in self.cross_layers:
            # Compute outer product-like interaction
            cross_term = dense(cross_x)
            cross_x = cross_x * cross_term + cross_x
        # Deep network: implicit feature interactions
        deep_x = x0
        for layer in self.deep_layers:
            deep_x = layer(deep_x)
        # Concatenate outputs from cross and deep networks
        combined = tf.concat([cross_x, deep_x], axis=-1)
        return self.output_layer(combined)

def train_model(model, dataset, loss_fn, optimizer, epochs=1):
    """
    Generic training loop.
    """
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for batch_x, batch_y in dataset:
            with tf.GradientTape() as tape:
                preds = model(batch_x, training=True)
                loss = loss_fn(batch_y, preds)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += loss.numpy()
            num_batches += 1
        print(f"Epoch {epoch+1}: average loss = {epoch_loss / num_batches:.4f}")

if __name__ == "__main__":
    # Create a dummy ranking dataset (replace with real data in practice)
    num_samples = 100
    input_dim = 10
    x = np.random.rand(num_samples, input_dim).astype(np.float32)
    # For a ranking task, target labels could be continuous scores
    y = np.random.rand(num_samples, 1).astype(np.float32)

    # Convert to tf.data.Dataset
    batch_size = 16
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)

    # Instantiate model, loss function and optimizer
    model = DCN(input_dim=input_dim, cross_layers=2, deep_units=[64, 32])
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Train the model
    train_model(model, dataset, loss_fn, optimizer, epochs=3)
