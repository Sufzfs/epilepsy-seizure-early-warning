# Federated Learning Edge Node Simulator (PyTorch FedAvg)
import copy
import torch
import torch.nn as nn

# 1. Lightweight Conv1D Model for Edge EEG Telemetry
class EdgeSeizureNet(nn.Module):
    def __init__(self):
        super(EdgeSeizureNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, stride=2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8 * 248, 2)  # Binary classification: Normal vs Pre-Ictal

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 2. Local Model Training on Edge Hardware Node
def train_local_node(model, dataloader, epochs=1, lr=0.001):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            
    return model.state_dict()

# 3. Federated Averaging (FedAvg) Server Weight Aggregation
def federated_averaging(global_model, local_state_dicts):
    """
    Aggregates local edge model weights via FedAvg:
    w_{t+1} = sum(w_k) / K
    """
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack(
            [local_dict[key].float() for local_dict in local_state_dicts], dim=0
        ).mean(dim=0)
    global_model.load_state_dict(global_dict)
    return global_model

if __name__ == "__main__":
    print("[+] Initializing Federated Edge Seizure Network...")
    
    # Initialize Global Server Model
    global_model = EdgeSeizureNet()
    
    # Simulate 2 Edge Nodes (Private Local Patient EEG Batches)
    node1_eeg = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(32, 1, 500), torch.randint(0, 2, (32,))),
        batch_size=8
    )
    node2_eeg = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(32, 1, 500), torch.randint(0, 2, (32,))),
        batch_size=8
    )
    
    # Local Training on Edge Nodes
    print("[+] Edge Node 1: Fine-tuning local model weights...")
    w1 = train_local_node(copy.deepcopy(global_model), node1_eeg)
    
    print("[+] Edge Node 2: Fine-tuning local model weights...")
    w2 = train_local_node(copy.deepcopy(global_model), node2_eeg)
    
    # Aggregate Model Weights via FedAvg
    global_model = federated_averaging(global_model, [w1, w2])
    
    print("[+] FedAvg Global Aggregation Complete! Zero raw EEG patient data transmitted.")
