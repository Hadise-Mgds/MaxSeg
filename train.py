import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
from models.u3net import U3Net
from utils.dataset import load_nrrd_volume, create_dataloaders

# Load configuration parameters from YAML file
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the NRRD image and label volumes
image_volumes, label_volumes = load_nrrd_volume(
    config["image_path"],
    config["label_path"],
    config["image_shape"]
)

# Create PyTorch DataLoaders for training and testing
train_loader, test_loader = create_dataloaders(
    image_volumes,
    label_volumes,
    batch_size=config["batch_size"]
)

# Initialize the 3D U-Net model and move it to the selected device
model = U3Net().to(device)

# Define binary cross-entropy loss with logits
loss_function = nn.BCEWithLogitsLoss()

# Set up the optimizer (Stochastic Gradient Descent with momentum and weight decay)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=config["lr"],
    momentum=0.9,
    weight_decay=1e-2
)

# Set random seed for reproducibility
torch.manual_seed(config["seed"])

# Training loop
for epoch in tqdm(range(config["epochs"])):
    print(f"Epoch {epoch + 1}/{config['epochs']}")

    # ----- Training Phase -----
    model.train()
    total_train_loss = 0.0
    for image_batch, label_batch in train_loader:
        image_batch, label_batch = image_batch.to(device), label_batch.to(device)

        # Forward pass
        prediction = model(image_batch)

        # Compute loss
        loss = loss_function(prediction, label_batch)

        # Backpropagation and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    average_train_loss = total_train_loss / len(train_loader)

    # ----- Evaluation Phase -----
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for image_batch, label_batch in test_loader:
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            prediction = model(image_batch)
            total_test_loss += loss_function(prediction, label_batch).item()

    average_test_loss = total_test_loss / len(test_loader)

    # Print training and testing loss for the current epoch
    print(f"Train Loss: {average_train_loss:.4f} | Test Loss: {average_test_loss:.4f}")
