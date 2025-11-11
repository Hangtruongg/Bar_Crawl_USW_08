import os
import pickle

# Paths
seq_results_path = 'results'
nonseq_results_path = 'results_nonseq'

all_results = {}

# Function to read all pkl files and store last validation loss
def load_results(path):
    results = {}
    for file in os.listdir(path):
        if file.endswith('.pkl'):
            with open(os.path.join(path, file), 'rb') as f:
                data = pickle.load(f)
                val_loss = data['val'][-1]  # last epoch validation loss
                results[file.replace('loss_','').replace('.pkl','')] = val_loss
    return results

# Load sequential and non-sequential results
all_results.update(load_results(seq_results_path))
all_results.update(load_results(nonseq_results_path))

# -----------------------------
# Best configuration per model type
# -----------------------------
best_per_model = {}

for name, val_loss in all_results.items():
    # Extract model type from file name (first part before _)
    model_type = name.split('_')[0]
    if model_type not in best_per_model or val_loss < best_per_model[model_type][1]:
        best_per_model[model_type] = (name, val_loss)

print("=== BEST CONFIG PER MODEL ===")
for model_type, (config_name, val_loss) in best_per_model.items():
    print(f"{model_type:10s} -> {config_name:50s} -> Val Loss: {val_loss:.6f}")

# -----------------------------
# Overall best
# -----------------------------
overall_best = min(best_per_model.values(), key=lambda x: x[1])
print(f"\n=== OVERALL BEST CONFIG ===\n{overall_best[0]} -> Val Loss: {overall_best[1]:.6f}")

# === BEST CONFIG PER MODEL ===
# BiLSTM     -> BiLSTM_seq100_hidden128-64_layers2_drop0.0_Adam_MSELoss_bs32 -> Val Loss: 0.001959
# GRU        -> GRU_seq500_hidden64-32_layers1_drop0.2_Adam_MSELoss_bs32 -> Val Loss: 0.002373
# LSTM       -> LSTM_seq500_hidden128-64_layers1_drop0.0_Adam_MSELoss_bs64 -> Val Loss: 0.002782
# Conv1D     -> Conv1D_hidden64-32_relu_Adam_drop0.1               -> Val Loss: 0.003624
# FeedForward -> FeedForward_hidden64-32_relu_Adam_drop0.1          -> Val Loss: 0.004550

# === OVERALL BEST CONFIG ===
# BiLSTM_seq100_hidden128-64_layers2_drop0.0_Adam_MSELoss_bs32 -> Val Loss: 0.001959