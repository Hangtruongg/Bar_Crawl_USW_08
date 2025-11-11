import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import itertools
import bc_data as bc  
from models.sequential import LSTMNet, GRUNet, BiLSTMNet

# -----------------------------
# Base settings
# -----------------------------
num_epochs = 20
input_size = 3  # x, y, z accelerometer
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

# -----------------------------
# Hyperparameter grids
# -----------------------------
seq_sizes = [50, 100, 500]
hidden_sizes_list = [[32, 16], [64, 32], [128, 64]]  # used for model naming
num_layers_list = [1, 2]
dropouts = [0.0, 0.2]
batch_sizes = [32, 64]
optimizers_list = ['Adam', 'SGD']
loss_functions_list = ['MSELoss', 'L1Loss']

# -----------------------------
# Sequential models
# -----------------------------
model_classes = {
    'LSTM': LSTMNet,
    'GRU': GRUNet,
    'BiLSTM': BiLSTMNet
}

# -----------------------------
# Dictionary to store best result per model
# -----------------------------
best_configs = {}

# -----------------------------
# Experiment loop
# -----------------------------
for model_name, ModelClass in model_classes.items():
    print(f"\n=== Experimenting with {model_name} ===")
    best_val_loss = float('inf')
    best_config_name = None

    # Loop through all hyperparameter combinations
    for seq_size, hidden_sizes, num_layers, dropout, batch_size, opt_name, loss_name in itertools.product(
        seq_sizes, hidden_sizes_list, num_layers_list, dropouts, batch_sizes, optimizers_list, loss_functions_list
    ):

        config_name = f"{model_name}_seq{seq_size}_hidden{'-'.join(map(str,hidden_sizes))}_layers{num_layers}_drop{dropout}_{opt_name}_{loss_name}_bs{batch_size}"
        print(f"\n--- Config: {config_name} ---")

        # Dataset and dataloader
        dataset = bc.BarCrawlDataset(seq_size)
        val_ratio = 0.2
        val_size = int(len(dataset) * val_ratio)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create model
        hidden_size = hidden_sizes[0]  # For LSTM/GRU/BiLSTM only the first hidden size is used
        model = ModelClass(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

        # Loss and optimizer
        criterion = getattr(nn, loss_name)()
        optimizer = getattr(optim, opt_name)(model.parameters(), lr=0.001)

        # History lists
        train_loss_history = []
        val_loss_history = []

        # Training loop
        for epoch in range(num_epochs):
            # Train
            model.train()
            running_train_loss = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.float()
                labels = labels.float().view(-1,1)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
            train_loss = running_train_loss / len(train_loader)
            train_loss_history.append(train_loss)

            # Validation
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.float()
                    labels = labels.float().view(-1,1)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()
            val_loss = running_val_loss / len(val_loader)
            val_loss_history.append(val_loss)

            print(f"Epoch {epoch+1}/{num_epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")

        # Save plot
        plt.figure(figsize=(8,5))
        plt.plot(range(1, num_epochs+1), train_loss_history, marker='o', label='Train Loss')
        plt.plot(range(1, num_epochs+1), val_loss_history, marker='s', label='Validation Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(config_name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = f"plots/{config_name}.png"
        plt.savefig(plot_path)
        plt.close()

        # Save loss history
        history_path = f"results/loss_{config_name}.pkl"
        with open(history_path, 'wb') as f:
            pickle.dump({'train': train_loss_history, 'val': val_loss_history}, f)

        # Track best config
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config_name = config_name

    best_configs[model_name] = (best_config_name, best_val_loss)
    print(f"\n*** Best config for {model_name}: {best_config_name} with Val Loss: {best_val_loss:.6f} ***")

# -----------------------------
# Summary of all best configs
# -----------------------------
print("\n=== BEST CONFIGS SUMMARY ===")
for model, (config, val_loss) in best_configs.items():
    print(f"{model:10s} -> {config:70s} -> Val Loss: {val_loss:.6f}")
