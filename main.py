import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from siamese.data import SiameseData
from torch.optim import Optimizer
import torch.nn as nn
from utils import plot_history

from siamese.model import SiameseNetwork
from siamese.data import SiameseData
from contrastive_loss import ContrastiveLoss

import numpy as np
import random
import wandb

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="ml-iris",

#     # track hyperparameters and run metadata
#     config={
#         "learning_rate": 0.02,
#         "architecture": "CNN",
#         "dataset": "CIFAR-100",
#         "epochs": 10,
#     }
# )


def train_loop(dataloader: DataLoader,
               model: nn.Module,
               loss_fn: nn.Module,
               optimizer: Optimizer,
               device: torch.device):
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    model.train()

    train_loss, accuracy = 0, 0
    for batch_index, (img1, img2, label) in enumerate(dataloader):
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        # forward
        output1, output2 = model(img1, img2)
        loss = loss_fn(output1, output2, label)
        loss.backward()
        train_loss += loss.item()
        # adam step
        optimizer.step()
        optimizer.zero_grad()

        eucledian_distance = nn.functional.pairwise_distance(output1, output2)
        accuracy += (eucledian_distance.argmax() ==
                     label).type(torch.float).sum().item()

        if batch_index % 100 == 0:
            loss, current = loss.item(), (batch_index + 1) * len(img1)
            print(f"loss: {loss:>7f}  [{((current / size) * 100):.2f}%]")

    train_loss /= num_batches
    accuracy /= size
    return train_loss, accuracy*100


def test_loop(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, accuracy = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for index, (img1, img2, label) in enumerate(dataloader):
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            output1, output2 = model(img1, img2)
            test_loss += loss_fn(output1, output2, label).item()
            eucledian_distance = nn.functional.pairwise_distance(
                output1, output2)
            accuracy += (eucledian_distance.argmax() ==
                         label).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy /= size
    return accuracy*100, test_loss


def train(train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          model: nn.Module,
          loss_fn: nn.Module,
          optimizer_fn: Optimizer,
          device: torch.device,
          learning_rate: float,
          epochs: int,
          name: str = 'model'):
    optimizer = optimizer_fn(model.parameters(), learning_rate)
    train_accuracies, train_losses, val_accuracies, val_losses = [], [], [], []
    
    best_val_loss = 1e5
    best_val_accuracy = .0
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss, train_accuracy = train_loop(train_dataloader, model,
                                                loss_fn, optimizer, device)
        val_accuracy, val_loss = test_loop(test_dataloader, model, loss_fn)

        # wandb.log({"train_accuracy": train_accuracy, 
        #            "train_loss": train_loss,
        #            "val_accuracy": val_accuracy,
        #            "val_loss": val_loss})
        
        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

        print(f"accuracy: {(train_accuracy):>0.1f}%, loss: {train_loss:>8f} \n" +
              f"val_accuracy: {(val_accuracy):>0.1f}%, val_loss: {val_loss:>8f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy

            print(f'Best model has val_accuracy:' +
                  f'{(best_val_accuracy):>0.1f}%, val_loss: {best_val_loss:>8f}')
            torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_fn': loss_fn,
                'learning_rate': learning_rate,
                'best_val_loss': best_val_loss,
                'best_val_accuracy': best_val_accuracy,
            }, 'model-checkpoint.pt')
    print("Training is done!")
    
    filename = f'{name}.png'
    plot_history(train_accuracies, val_accuracies, train_losses, val_losses, filename)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == '__main__':
    # garante a reproducibilidade
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train_dir = "train"
    test_dir = "test"
    train_csv_path = "train_data.csv"
    test_csv_path = "test_data.csv"

    IMG_HEIGHT = 105
    IMG_WIDTH = 105
    # Transformações usadas no dataset
    ds_transforms = transforms.Compose([transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
                                        transforms.ToTensor()])

    # Set datasets
    train_ds = SiameseData(train_csv_path, train_dir, transform=ds_transforms)
    test_ds = SiameseData(test_csv_path, test_dir, transform=ds_transforms)

    # Set dataloaders
    BATCH_SIZE = 32
    train_dataloader = DataLoader(train_ds,
                                  shuffle=True,
                                  num_workers=8,
                                  pin_memory=True,
                                  batch_size=BATCH_SIZE,
                                  worker_init_fn=seed_worker,
                                  generator=torch.Generator().manual_seed(RANDOM_SEED))
    test_dataloader = DataLoader(test_ds,
                                 shuffle=False,
                                 num_workers=8,
                                 pin_memory=True,
                                 batch_size=BATCH_SIZE,
                                 worker_init_fn=seed_worker,
                                 generator=torch.Generator().manual_seed(RANDOM_SEED))

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Initiating cuda device {torch.cuda.get_device_name(0)}')
    model = SiameseNetwork(IMG_HEIGHT, IMG_WIDTH)
    model = model.to(device)

    epochs = 5
    learning_rate = 1E-3
    loss_fn = ContrastiveLoss()
    optimizer_fn = torch.optim.Adam

    # train(train_dataloader,
    #       test_dataloader,
    #       model,
    #       loss_fn,
    #       optimizer_fn,
    #       device,
    #       learning_rate,
    #       epochs)
    model = torch.load('model-checkpoint.pt')
    
    # train_accuracy, train_loss = test_loop(train_dataloader, model, loss_fn)
    # val_accuracy, test_loss = test_loop(test_dataloader, model, loss_fn)

    # print(train_accuracy, train_loss)
    # print(val_accuracy, test_loss)
