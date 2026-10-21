import torch
from tqdm import tqdm
import torch.optim as optim

from src.utils.model_registry import ModelRegistryManager
from src.utils.device import get_device
from src.utils.get_folders_utils import get_dataset_name

def train_one_epoch(model, train_loader, device, optimizer):
    model.train()
    train_loss = 0

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        batch = {"x": x, "y": y}

        output = model(x)
        loss = model.loss(batch, output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    
    return train_loss

def validate(model, val_loader, device):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x, y in val_loader:

            x = x.to(device)
            y = y.to(device)

            batch = {"x": x, "y": y}

            output = model(x)
            loss = model.loss(batch, output)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    return val_loss

def train(model, train_loader, val_loader, config):

    registry = ModelRegistryManager(config)

    device = get_device()
    model.to(device)
    
    epochs = config["training"]["epochs"]
    patience = config["training"]["patience"]
    lr = config["training"]["lr"]

    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    patience_counter = 0

    model_name = config["training"]["model"]
    dataset_name = get_dataset_name(config)
    print(f"\n {model_name} training on {dataset_name} started...\n")

    for epoch in tqdm(range(epochs)):

        train_loss = train_one_epoch(model, train_loader, device, optimizer)
        val_loss = validate(model, val_loader, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        # Save LAST model
        registry.save_last(
            model=model,
            val_loss=val_loss,
            epoch=epoch + 1
        )

        # Save BEST model
        improved = registry.save_best_if_improved(
            model=model,
            val_loss=val_loss,
            epoch=epoch + 1
        )

        if improved:
            patience_counter = 0
            print("New best model saved")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\n Early stopping at epoch {epoch+1}")
            break

    print("\n Training completed\n")