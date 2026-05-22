import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import copy
import os
import matplotlib.pyplot as plt

# -----------------------------
# Dataset class
# -----------------------------
class ClientDataset(Dataset):
    def __init__(self, file_path):
        data = pd.read_csv(file_path)
        y = data.iloc[:, -1]

        # Convert string labels to integers
        if y.dtype == 'object':
            y, _ = pd.factorize(y)

        X = data.iloc[:, :-1]
        X = pd.get_dummies(X)
        X = X.astype("float32")

        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -----------------------------
# Import FL components
# -----------------------------
from model import GlobalModel
from server import Server
from client import Client

# -----------------------------
# Settings
# -----------------------------
NUM_ROUNDS = 10        # Total federated rounds
BATCH_SIZE = 32
LR = 0.01
MU = 0.01             # FedProx proximal term
DEVICE = "cpu"        # Use "cuda" if GPU is available

# -----------------------------
# Client dataset paths
# -----------------------------
client_paths = [
    r"C:\Users\SHRUTI\OneDrive\Documents\majorproject\client_datasets_3_clients\client_1.csv",
    r"C:\Users\SHRUTI\OneDrive\Documents\majorproject\client_datasets_3_clients\client_2.csv",
    r"C:\Users\SHRUTI\OneDrive\Documents\majorproject\client_datasets_3_clients\client_3.csv"
]

# -----------------------------
# Initialize server and clients
# -----------------------------
sample_dataset = ClientDataset(client_paths[0])
INPUT_DIM = sample_dataset.X.shape[1]
NUM_CLASSES = 2

# Initialize global model on server
global_model = GlobalModel(INPUT_DIM, NUM_CLASSES)
server = Server(global_model)

clients = []
client_sizes = []
client_test_datasets = []  # For per-client evaluation

for i, path in enumerate(client_paths):
    # Load full client dataset
    full_dataset = ClientDataset(path)
    # Split 80% train, 20% test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    clients.append(Client(client_id=i, model=global_model, train_loader=loader, lr=LR, device=DEVICE))
    client_sizes.append(len(train_dataset))
    client_test_datasets.append(test_dataset)  # Save test part

# -----------------------------
# Evaluation function
# -----------------------------
def evaluate_model(model, datasets):
    """
    Evaluate model on a list of datasets (can be Subset or ClientDataset)
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for dataset in datasets:
            loader = DataLoader(dataset, batch_size=64, shuffle=False)
            for X, y in loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                outputs = model(X)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
    return correct / total


# -----------------------------
# Hybrid Federated Learning Loop
# FedAvg + StatAvg + FedProx combined
# -----------------------------
global_acc_history = []
per_client_acc_history = [[] for _ in range(len(clients))]

for rnd in range(1, NUM_ROUNDS + 1):
    print(f"\n=== Federated Round {rnd} ===")

    # 1️⃣ Server sends global weights to clients
    global_weights = server.get_weights()
    client_weights = []

    for client in clients:
        client.set_weights(global_weights)

        # 2️⃣ Client trains locally with FedProx
        client.train(epochs=1, mu=MU)

        # 3️⃣ Clients send updated weights back to server
        client_weights.append(client.get_weights())

    # 4️⃣ Server aggregates weights (FedAvg + StatAvg hybrid)
    new_weights = copy.deepcopy(client_weights[0])
    total_samples = sum(client_sizes)

    for key in new_weights.keys():
        # FedAvg part: weighted by dataset sizes
        fedavg_part = sum(client_weights[i][key] * (client_sizes[i] / total_samples)
                          for i in range(len(clients)))

        # StatAvg part: normalize updates by magnitude
        norms = [torch.norm(client_weights[i][key] - global_weights[key]) for i in range(len(clients))]
        total_norm = sum(norms) + 1e-10
        statavg_part = sum((client_weights[i][key] - global_weights[key]) * (norms[i] / total_norm)
                           for i in range(len(clients))) + global_weights[key]

        # Combine FedAvg + StatAvg
        new_weights[key] = (fedavg_part + statavg_part) / 2

    # 5️⃣ Update global model
    server.set_weights(new_weights)

    # 6️⃣ Evaluate global model on all clients' test datasets
    global_acc = evaluate_model(server.global_model, client_test_datasets)
    global_acc_history.append(global_acc)
    print(f"Global Accuracy after round {rnd}: {global_acc:.4f}")

    # 7️⃣ Evaluate per-client accuracy
    for i, test_data in enumerate(client_test_datasets):
        acc = evaluate_model(server.global_model, [test_data])
        per_client_acc_history[i].append(acc)
        print(f"Client {i+1} test accuracy: {acc:.4f}")

# -----------------------------
# Save global model
# -----------------------------
os.makedirs("saved_models", exist_ok=True)
save_path = "saved_models/global_model_hybrid.pth"
torch.save(server.global_model.state_dict(), save_path)
print(f"\n✅ Global model saved at {save_path}")

# -----------------------------
# Plot global accuracy over rounds
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(range(1, NUM_ROUNDS + 1), global_acc_history, marker='o', label='Global Accuracy')
for i in range(len(clients)):
    plt.plot(range(1, NUM_ROUNDS + 1), per_client_acc_history[i], marker='x', linestyle='--', label=f'Client {i+1} Accuracy')
plt.xlabel('Federated Round')
plt.ylabel('Accuracy')
plt.title('Hybrid Federated Learning Accuracy')
plt.legend()
plt.grid(True)
plt.show()
