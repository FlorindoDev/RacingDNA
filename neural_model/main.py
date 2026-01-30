import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # necessario per il 3D
from auto_encoder import AutoEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401




path = "..\data\dataset\\normalized_dataset.npz"  # cambia con il tuo file
d = np.load(path, allow_pickle=True)

data = d["data"]
mask = d["mask"]

# print("Data shape:", data.shape)
# print("Mask shape:", mask.shape)
# print("First data sample (first 50 elements):")
# print(data[0][:50])
# print(mask[0][:50])



model = AutoEncoder(data.shape[1], latent_dim=32)
# model.train(
#         optimizer=torch.optim.SGD(model.parameters(), lr=0.001),
#         epochs=1,
#         input=data,
#         mask=mask
#     )   

# torch.save(model.encoder.state_dict(), ".\\Pesi\\encoder.pth")

model.encoder.load_state_dict(torch.load(".\\Pesi\\encoder.pth", map_location="cpu"))


avg = []
with torch.no_grad():

    for elem in data[:1000]:
        z = model.forward(torch.tensor(np.atleast_1d(elem), dtype=torch.float32),True)
        avg.append(z.cpu().numpy())

Z = np.array(avg)  # shape: (N, 32)



# Riduzione a 3 dimensioni
Z3 = PCA(n_components=3).fit_transform(Z)

# Plot 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(Z3[:, 0], Z3[:, 1], Z3[:, 2])
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.show()