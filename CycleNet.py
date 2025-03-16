import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv(r"input.csv")
df = df.iloc[:, 0]

scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(df.values.reshape(-1, 1))

W = 24
D = 1


class CycleNet(nn.Module):
    def __init__(self, W, D):
        super(CycleNet, self).__init__()
        self.W = W
        self.D = D
        self.cycleQueue = nn.Parameter(torch.zeros(W, D))
        self.residual_model = nn.Linear(D, D)

    def forward(self, t):
        idx = t % self.W
        C_t = self.cycleQueue[idx, :]
        return C_t


model = CycleNet(W, D)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()

    t = torch.tensor(epoch % W, dtype=torch.long)

    X_t = torch.tensor(df[epoch % len(df)].astype(float), dtype=torch.float32).unsqueeze(0)

    C_t = model(t)

    loss = loss_fn(C_t, X_t)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

Q_learned = model.cycleQueue.detach().numpy().flatten()

Q_reshaped = Q_learned.reshape(-1, 1)

Q_inverse_transformed = scaler.inverse_transform(Q_reshaped)

print(Q_inverse_transformed)

plt.figure(figsize=(8, 4))
plt.plot(Q_inverse_transformed, marker='o', linestyle='-')
plt.xlabel()
plt.ylabel()
plt.title()
plt.grid()
plt.show()
