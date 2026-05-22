import torch
import copy
from torch.utils.data import DataLoader, random_split
from model import GlobalModel
from server import Server
from client import Client
import pandas as pd


# Same dataset logic you used before
class ClientDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        data = pd.read_csv(file_path)
        y = data.iloc[:, -1]

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


class FederatedEngine:
    def __init__(self, client_paths, device="cpu"):
        self.device = device

        # Determine input size
        sample_data = pd.read_csv(client_paths[0])
        X = pd.get_dummies(sample_data.iloc[:, :-1])
        input_dim = X.shape[1]
        num_classes = 2

        self.global_model = GlobalModel(input_dim, num_classes).to(device)
        self.server = Server(self.global_model)

        self.clients = []
        self.client_sizes = []
        self.test_datasets = []

        for i, path in enumerate(client_paths):
            full_dataset = ClientDataset(path)

            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

            loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            client = Client(i, self.global_model, loader, lr=0.01, device=device)
            self.clients.append(client)
            self.client_sizes.append(len(train_dataset))
            self.test_datasets.append(test_dataset)

        self.round = 0
        self.history = []

    def evaluate(self):
        self.global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for dataset in self.test_datasets:
                loader = DataLoader(dataset, batch_size=64)
                for X, y in loader:
                    X, y = X.to(self.device), y.to(self.device)
                    pred = torch.argmax(self.global_model(X), dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
        return correct / total

    def train_one_round(self, mu=0.01):
        self.round += 1

        global_weights = self.server.get_weights()
        client_weights = []

        for client in self.clients:
            client.set_weights(global_weights)
            client.train(epochs=1, mu=mu)
            client_weights.append(client.get_weights())

        # FedAvg aggregation
        new_weights = copy.deepcopy(client_weights[0])
        total_samples = sum(self.client_sizes)

        for key in new_weights:
            new_weights[key] = sum(
                client_weights[i][key] * (self.client_sizes[i] / total_samples)
                for i in range(len(self.clients))
            )

        self.server.set_weights(new_weights)

        acc = self.evaluate()
        self.history.append({"round": self.round, "accuracy": acc})

        return acc
