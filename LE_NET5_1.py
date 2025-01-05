import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp as amp

import time

import torchvision
import torchvision.transforms as transforms

from torchsummary import summary

from model import LeNet5

# Do we need this transformation, what does it do ??? Maybe we can consider normalization hypper-parameter
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Grayscale(num_output_channels=1)
])

def load_data():
    # This download the train and test data in CIFAR-10. They are not in the repositorie while they are stored in the folder data in the root
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=1, pin_memory=True, prefetch_factor=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=1, pin_memory=True, prefetch_factor=2)
    return trainloader, testloader

# Define the classes in CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# our dataset has 10 classes
num_classes = len(classes)

def train_model(model, trainloader, criterion, optimizer, device, epochs=10, verbose='on'):
    model.train()
    torch.backends.cudnn.benchmark = True

    total_time = 0

    for epoch in range(epochs):
        running_loss = 0.0
        start = time.time()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_time = time.time() - start
        total_time += epoch_time

        if verbose=='on' or (verbose=='partial' and ((epoch+1)%5==0 or epoch==0)):
            print(f"Epoch {epoch+1}/{epochs}, "
                f"Loss: {running_loss/len(trainloader):.4f}, "
                f"Time: {epoch_time:.2f}s, "
                f"Memory: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

    return 0

def test_model(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")
    return correct / total

if __name__ == '__main__':
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print(f"Using {device} device")
    
    trainloader, testloader = load_data()
    model = LeNet5()
    if device == "cuda":
        model = model.type(torch.cuda.FloatTensor)
        torch.cuda.empty_cache()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    summary(model,(1,32,32),64)
    train_model(model, trainloader, criterion, optimizer, device, epochs=1)
    test_model(model, testloader, device)