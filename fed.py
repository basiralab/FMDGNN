import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
from collections import OrderedDict

from torch.utils.data import Dataset, DataLoader
from losses import reconstruction_loss


class Client:
    def __init__(self, local_model, local_optimizer, local_data,
                 local_epochs, batch_size, device, name):
        self.model = local_model
        self.optimizer = local_optimizer
        self.local_views = [int(view) for view in self.model.decoders.keys()]
        self.local_loader = DataLoader(local_data, batch_size=batch_size, shuffle=True)
        self.local_epochs = local_epochs
        self.device = device
        self.name = name

    def get_weights(self):
        return self.model.state_dict()

    def update_weights(self):
        train_loss_log = []

        for epoch in range(self.local_epochs):
            train_loss = 0
            for src, tgt1, tgt2, tgt3, tgt4, tgt5 in self.local_loader:
                torch.cuda.empty_cache()
                tgts = [tgt1, tgt2, tgt3, tgt4, tgt5]
                local_tgts = [tgts[int(view) - 1] for view in self.model.decoders.keys()]
                edge_idx = torch.Tensor([[i for i in range(src.shape[0])]] * 2).long().to(self.device)

                # train the local model
                self.model.train()
                self.optimizer.zero_grad()

                outputs = self.model(src, edge_idx)

                # reconstruction loss
                recon_loss = sum(
                    [reconstruction_loss(outputs[str(view)], tgt) for view, tgt in zip(self.local_views, local_tgts)])

                loss = recon_loss

                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            train_loss_log.append(train_loss / len(self.local_loader))
            print(f"Client {self.name}  Local Epoch: {epoch + 1}  Loss: {train_loss / len(self.local_loader)}")

        return train_loss_log

    def set_weights(self, state_dict):
        weights = self.get_weights()
        for k, v in state_dict.items():
            if k in weights.keys():
                weights.update({k: v})
        self.model.load_state_dict(weights)


class Server:
    def __init__(self, clients: list[Client], global_model, aggr="FedAvg"):
        self.clients = clients
        self.global_model = global_model
        self.aggr = aggr
        self.mu = 0.1

    def broadcast_weights(self):
        state_dict = self.averaging_weights()
        for client in self.clients:
            client.set_weights(state_dict)

    def get_weights(self):
        return self.global_model.state_dict()

    def set_weights(self, state_dict):
        weights = self.get_weights()
        for k, v in state_dict.items():
            if k in weights.keys():
                weights.update({k: v})
        self.global_model.load_state_dict(weights)

    def averaging_weights(self):
        client_weights = [client.get_weights() for client in self.clients]
        average_weights = OrderedDict()

        keys = set()
        for client_w in client_weights:
            keys.update(set(client_w.keys()))

        for key in keys:
            # federate both the encoder and decoders
            curr_weight = [client[key] for client in client_weights if key in client]
            if not curr_weight: continue

            average_weights[key] = torch.stack(curr_weight).mean(dim=0)
            self.set_weights(average_weights)

        return average_weights

    def federate(self, rounds):
        losses = [[] for _ in range(len(self.clients))]

        for r in range(rounds):
            for j, client in enumerate(self.clients):
                client_loss = client.update_weights()
                losses[j].extend(client_loss)

            self.broadcast_weights()

        return losses




