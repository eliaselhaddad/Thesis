import os
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ModelUtilities:

    def load_data(self, file_path: str) -> pd.DataFrame:
        logging.info(f"Loading data from {file_path}")
        return pd.read_csv(file_path)

    def filter_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data) > 110:
            logging.info("Data exceeds length limit and will be excluded")
            return None
        data = data.drop(columns=["g_force"])
        return data.dropna()

    def separate_features_labels(self, data: pd.DataFrame) -> tuple[np.ndarray, int]:
        label = data["fall_state"].iloc[
            0
        ]  # Assuming all rows have the same label in a single file
        features = data.drop(columns=["fall_state"]).to_numpy()
        return features, label

    def pad_sequences(self, sequences: list) -> np.ndarray:
        logging.info("Padding sequences to the same length")
        return pad_sequences(
            sequences, dtype="float32", padding="post", truncating="post"
        )

    def scale_sequences(self, sequences: np.ndarray) -> np.ndarray:
        logging.info("Scaling sequences using MinMaxScaler")
        scaler = MinMaxScaler()
        scaled_sequences = scaler.fit_transform(
            sequences.reshape(-1, sequences.shape[-1])
        ).reshape(sequences.shape)
        return scaled_sequences

    def split_data(self, sequences: np.ndarray, labels: np.ndarray) -> tuple:
        logging.info(
            f"Attempting to split data with {len(sequences)} sequences and {len(labels)} labels."
        )
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def load_and_prepare_data(self, directory: str) -> tuple:
        all_features = []
        all_labels = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            data = self.load_data(file_path)
            filtered_data = self.filter_data(data)
            if filtered_data is not None:
                features, label = self.separate_features_labels(filtered_data)
                all_features.append(features)
                all_labels.append(label)  # Collecting one label per file
        padded_features = self.pad_sequences(all_features)
        scaled_features = self.scale_sequences(padded_features)
        return self.split_data(scaled_features, np.array(all_labels))


def main():
    utilities = ModelUtilities()
    directory = "data/seq/"
    X_train, X_val, X_test, y_train, y_val, y_test = utilities.load_and_prepare_data(
        directory
    )
    logging.info("Data loading and preprocessing complete. Ready for model training.")


if __name__ == "__main__":
    main()
