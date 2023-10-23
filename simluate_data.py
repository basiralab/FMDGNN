import numpy as np
import os

from config import N_SUBJECTS


def get_simulate_data():
    # Simulate data
    features = np.random.randn(N_SUBJECTS, 595)

    # Save to numpy files
    save_path = f'./simulated_dataset'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save("simulated_dataset/simulate_data.npy", features)


if __name__ == '__main__':
    get_simulate_data()
