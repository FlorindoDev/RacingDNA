import time
import torch
import torch.nn as nn
import numpy as np

from torch import optim
import matplotlib.pyplot as plt
import csv





print("torch:", torch.__version__)
print("torch.version.hip:", torch.version.hip)
print("cuda.is_available (ROCm usa questa API):", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    print("device 0:", torch.cuda.get_device_name(0))
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")



class AutoEncoder(nn.Module):
    def __init__(self, input_dim=605, latent_dim=32):
        super().__init__()
        self.losses = []
        self.input_dim= input_dim

        # Encoder: x -> z
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        # Decoder: z -> x_hat
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def add_padding(self,x):
        #for _ in range(len(x),self.input_dim):
        pass



    def train(self,loss_function,optimizer,epochs,data_set, bach_size=32):
        outputs = []

        # tensor_transform = transforms.ToTensor()
        # dataset = datasets.MNIST(root="./data", train=True, download=True, transform=tensor_transform)
        # loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)



        


        # for epoch in range(epochs):
        #     for images, _ in loader:
        #         images = images.view(-1, 28 * 28).to(device)
                
        #         reconstructed = model(images)
        #         loss = loss_function(reconstructed, images)
                
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
                
        #         self.losses.append(loss.item())
            
        #     outputs.append((epoch, images, reconstructed))
        #     print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")






model = AutoEncoder()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

model.to(device)


# with open("p.csv", newline="", encoding="utf-8") as f:
#     reader = csv.DictReader(f)   # usa la prima riga come nomi colonna
#     rows = list(reader)

# print(rows[0])   # prima riga come dict
X = np.loadtxt("p.csv", delimiter=",", skiprows=1)  # skip header
len(X[0])
X_vet = X.flatten()
X_tensor = torch.tensor(X_vet, dtype=torch.float32)
print(X_tensor)

# model.train()

plt.style.use('fivethirtyeight')
plt.figure(figsize=(8, 5))
plt.plot(model.losses, label='Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()




