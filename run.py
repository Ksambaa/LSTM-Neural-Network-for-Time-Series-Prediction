__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import os
import json
import time
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model

def plot_results(predicted_data, true_data, dates=None):
    """Plot the predicted vs actual water debit"""
    plt.figure(figsize=(15, 7))
    plt.plot(true_data, label='True Data', linewidth=2, alpha=0.7)
    plt.plot(predicted_data, label='Prediction', linewidth=2, alpha=0.7)
    plt.title('Water Debit Prediction', fontsize=14)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Debit', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

def main():
    config = json.load(open('config.json', 'r'))

    if not os.path.exists(config['model']['save_dir']):
        os.makedirs(config['model']['save_dir'])

    data = DataLoader(
        os.path.join('data', config['data']['filename']),
        config['data']['train_test_split'],
        config['data']['columns']
    )

    model = Model()
    model.build_model(config)

    x, y = data.get_train_data(
        seq_len=config['data']['sequence_length'],
        normalise=config['data']['normalise']
    )

    # Train the model
    steps_per_epoch = math.ceil((data.len_train - config['data']['sequence_length']) / config['training']['batch_size'])
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=config['data']['sequence_length'],
            batch_size=config['training']['batch_size'],
            normalise=config['data']['normalise']
        ),
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_dir=config['model']['save_dir']
    )

    # Make predictions
    x_test, y_test = data.get_test_data(
        seq_len=config['data']['sequence_length'],
        normalise=config['data']['normalise']
    )
    predictions = model.predict_sequences_multiple(
        x_test, 
        config['data']['sequence_length'],
        config['data']['sequence_length']
    )

    # Plot results
    plot_results(predictions, y_test)

if __name__ == '__main__':
    main()