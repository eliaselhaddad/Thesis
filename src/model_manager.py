import argparse
import logging
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from model_utilities import ModelUtilities

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ModelManager:
    def __init__(self, model_type):
        self.model_type = model_type
        self.model = None
        self.utilities = ModelUtilities()

    def load_data(self):
        logging.info("Loading and preparing data")
        # Using the function to load and prepare data directly
        return self.utilities.load_and_prepare_data("data/seq/")

    def build_model(self, input_shape):
        logging.info(f"Building {self.model_type} with input shape {input_shape}")
        if self.model_type == "base_model":
            self.model = self.create_base_model(input_shape)
            logging.info("Base model created.")
        elif self.model_type == "pruned_model":
            self.model = self.create_pruned_model(input_shape)
            logging.info("Pruned model created.")
        else:
            logging.error(f"Model type {self.model_type} is not supported.")
            raise ValueError(f"Model type {self.model_type} is not supported.")

    def train_model(self, train_data, train_labels, val_data, val_labels):
        if not self.model:
            raise Exception("Model has not been built. Call build_model first.")
        logging.info(f"Training the {self.model_type}")
        history = self.model.fit(
            train_data,
            train_labels,
            epochs=100,
            validation_data=(val_data, val_labels),
            callbacks=[EarlyStopping(patience=10)],
        )
        return history

    def save_model(self):
        path = f"models/{self.model_type}"
        if not os.path.exists(path):
            os.makedirs(path)
        model_path = os.path.join(path, f"{self.model_type}.keras")
        self.model.save(model_path)
        logging.info(f"Model saved at {model_path}")

    def create_base_model(self, input_shape, l2_lambda=0.01):
        """Build a LSTM model for the given input shape with L2 regularization."""
        model = Sequential(
            [
                LSTM(128, input_shape=input_shape, return_sequences=True,
                    kernel_regularizer=l2(l2_lambda)),  
                Dropout(0.5),
                LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_lambda)),  
                Dropout(0.3),
                LSTM(32, kernel_regularizer=l2(l2_lambda)),  
                Dense(32, activation="relu", kernel_regularizer=l2(l2_lambda)),  
                Dropout(0.2),
                Dense(1, activation="sigmoid", kernel_regularizer=l2(l2_lambda)),  
            ]
        )
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model

    def create_pruned_model(self, input_shape):
        # Define how to construct the pruned model here
        pass

    def evaluate_model(self, test_data, test_labels):
        logging.info(f"Evaluating the {self.model_type}")
        results = self.model.evaluate(test_data, test_labels)
        logging.info(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")
        return results

    def plot_learning_curves(self, history):
        epochs = range(1, len(history.history["accuracy"]) + 1)
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, history.history["accuracy"], "bo-", label="Training Acc")
        plt.plot(epochs, history.history["val_accuracy"], "ro-", label="Validation Acc")
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, history.history["loss"], "bo-", label="Training Loss")
        plt.plot(epochs, history.history["val_loss"], "ro-", label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        # Save the plot
        plot_path = self.get_plot_path()
        plt.savefig(plot_path)
        logging.info(f"Plot saved at {plot_path}")
        # plt.close()  # Close the plot after saving to release memory

    def get_plot_path(self):
        path = f"plots/{self.model_type}"
        if not os.path.exists(path):
            os.makedirs(path)
        return os.path.join(path, f"{self.model_type}_training_validation_curves.png")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train different models with specified techniques."
    )
    parser.add_argument(
        "--model_type",
        choices=["base_model", "pruned_model"],
        required=True,
        help="Type of model to train",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()

    manager = ModelManager(args.model_type)
    train_data, val_data, test_data, train_labels, val_labels, test_labels = (
        manager.load_data()
    )

    # Ensure the model is built with the correct input shape
    manager.build_model(train_data.shape[1:])  # Excluding batch size dimension
    history = manager.train_model(train_data, train_labels, val_data, val_labels)
    manager.save_model()

    manager.evaluate_model(test_data, test_labels)
    manager.plot_learning_curves(history)


if __name__ == "__main__":
    main()
