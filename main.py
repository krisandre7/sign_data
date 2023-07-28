import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from siamese.data import SiameseData
from torch.optim import Optimizer
import torch.nn as nn

from siamese.model import SiameseNetwork
from siamese.data import SiameseData
from contrastive_loss import ContrastiveLoss

def train_loop(dataloader: DataLoader, 
          model: nn.Module, 
          loss_fn: nn.Module,
          optimizer: Optimizer, 
          device: torch.device):

    size = len(dataloader.dataset)
    model.train()
    for batch_index, (img1, img2, label) in enumerate(dataloader):
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        
        #forward
        output1, output2 = model(img1, img2)
        loss_contrastive = loss_fn(output1, output2, label)
        loss_contrastive.backward()
        #adam step  
        optimizer.step()
        optimizer.zero_grad()
        
        if batch_index % 100 == 0:
            loss_contrastive, current = loss_contrastive.item(), (batch_index + 1) * len(img1)
            print(f"loss_contrastive: {loss_contrastive:>7f}  [{(current / size):.2f}%]")
    
    return model

def test_loop(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for index, (img1, img2, label) in enumerate(dataloader):
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            output1, output2 = model(img1, img2)
            test_loss += loss_fn(output1, output2, label).item()
            eucledian_distance = nn.functional.pairwise_distance(output1, output2)
            print(eucledian_distance.item(), label)
            return
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # print(label)

        # print("Predicted Eucledian Distance:-",eucledian_distance.item())
        # print("Actual Label:-",label)
    # test_loss /= num_batches
    # correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def train(train_dataloader: DataLoader, 
          test_dataloader: DataLoader, 
          model: nn.Module, 
          loss_fn: nn.Module, 
          optimizer_fn: Optimizer, 
          device: torch.device, 
          learning_rate: float,
          epochs: int):
    optimizer = optimizer_fn(model.parameters(), learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        new_model = train_loop(train_dataloader, model, loss_fn, optimizer, device)
        torch.save(new_model, 'nu_model.pt')
        test_loop(test_dataloader, model, loss_fn)
        
        # TODO: Implementar checkpoint
        # PATH = "model.pt"
        # torch.save({
        #     'epoch': EPOCH,
        #     'model_state_dict': net.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': LOSS,
        #     }, PATH)
    print("Done!")

if __name__ == '__main__':
    train_dir = "sign_data/train"
    test_dir = "sign_data/test"
    train_csv_path = "sign_data/train_data.csv"
    test_csv_path =  "sign_data/test_data.csv"

    # Transformações usadas no dataset 
    ds_transforms = transforms.Compose([transforms.Resize((105,105)),
                                    transforms.ToTensor()])

    # Set datasets
    train_ds = SiameseData(train_csv_path,train_dir,transform=ds_transforms)
    test_ds = SiameseData(test_csv_path,test_dir, transform=ds_transforms)

    # Set dataloaders
    train_dataloader = DataLoader(train_ds, 
                                  shuffle=True,
                                  num_workers=8,
                                  pin_memory=True,
                                  batch_size=32)
    test_dataloader = DataLoader(test_ds,
                                 shuffle=False,
                                 num_workers=8,
                                 pin_memory=True,
                                 batch_size=32) 

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNetwork()
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
    
    model = torch.load('nu_model.pt')
    test_loop(test_dataloader, model, loss_fn)
    
    