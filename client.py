import torch
import copy

class Client:
    def __init__(self, client_id, model, train_loader, lr=0.01, device="cpu"):
        self.client_id = client_id
        self.device = device
        # Make a local copy of the model
        self.model = copy.deepcopy(model).to(self.device)
        self.train_loader = train_loader
        self.lr = lr

    def set_weights(self, weights):
        """Set local model weights from global model"""
        self.model.load_state_dict(weights)

    def get_weights(self):
        """Return a copy of local model weights"""
        return copy.deepcopy(self.model.state_dict())

    def train(self, epochs=1, mu=0.0):
        """
        Local training with optional FedProx proximal term
        :param epochs: Number of local epochs
        :param mu: FedProx coefficient, 0 disables FedProx
        """
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()

        # Save global weights for FedProx
        global_weights = copy.deepcopy(self.model.state_dict())

        for _ in range(epochs):
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(x)
                loss = criterion(outputs, y)

                # -----------------------------
                # FedProx proximal term
                # ||w_local - w_global||^2
                # -----------------------------
                if mu > 0.0:
                    prox_loss = 0.0
                    for name, param in self.model.named_parameters():
                        prox_loss += torch.norm(param - global_weights[name]) ** 2
                    loss += (mu / 2) * prox_loss

                loss.backward()
                optimizer.step()
