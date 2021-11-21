# !pip install jupyterthemes --quiet
from jupyterthemes import jtplot

jtplot.style(theme="monokai", context="notebook", ticks=True, grid=False)

import importlib
import os
import sys

sys.path.append(os.path.abspath(os.path.join("..", "..")))

from random import randint

import matplotlib.image as m_image
import matplotlib.pyplot as plt
import numpy as np

# +
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# -

np.random.seed(666)

from IPython.lib.deepreload import reload

# Load all the code we have in the module (and reload if needed)
import config

# %load_ext autoreload
# %autoreload 2


Config = config.Config

from src.dataloaders.birds import generate_file_dataset

# ## Data Loading

train_files = generate_file_dataset("train")
print(train_files.shape)
train_files.head(5)

print(f"We currently have {train_files['label'].unique().shape[0]} classes")
train_files["label"].value_counts().head(20).plot.barh(title="Distribution of the different labels")

plt.title("Distribution of the number of photos per class")
sns.histplot(x=train_files["label"].value_counts(), bins=50, kde=True)

# +
# Display some examples
W, H = 2, 4

random_indexes = [randint(0, train_files.shape[0]) for _ in range(W * H)]

fig, axes = plt.subplots(W, H, figsize=(10, 6))

for i in range(W):
    for j in range(H):
        ax = axes[i, j]
        row = train_files.iloc[random_indexes[i * H + j]]

        ax.imshow(m_image.imread(row["file"]))
        ax.set_title(row["label"])
plt.tight_layout()
# -

# ## Loading the data

from src.dataloaders.birds import BirdsDataset, BirdsDatasetTypes
from src.models.BirdsScratchCnn import BirdsScratchCnn as net

EPOCHS = 2
BATCH_SIZE = 20
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net.to(device=DEVICE)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

train_set = BirdsDataset("train", transform=transform)
test_set = BirdsDataset("test", transform=transform)

# +
trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
testloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    net.parameters(),
    lr=5e-3,
)

# +
for epoch in range(EPOCHS):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in tqdm(
        enumerate(trainloader, 0), total=len(train_set) // BATCH_SIZE, desc=f"EPOCH {epoch}"
    ):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print("Finished Training")
torch.save(net.state_dict(), os.path.join(Config.ROOT_DIRECTORY, "models", "birds", "ScratchModel"))
# -

# Check the prediction of the model on the test set
predictions = []
truth = []
with torch.no_grad():
    for i, data in tqdm(
        enumerate(testloader, 0), total=len(test_set) // BATCH_SIZE, desc=f"Compute predictions"
    ):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = net(inputs)
        truth.extend(labels.detach().tolist())
        predictions.extend(outputs.argmax(-1).detach().tolist())


from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

print(classification_report(truth, predictions))

ConfusionMatrixDisplay(confusion_matrix(truth, predictions)).plot()
