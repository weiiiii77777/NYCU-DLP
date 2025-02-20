import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from dataloader import BufferflyMothLoader
from VGG19 import VGG
from ResNet50 import ResNet
import matplotlib.pyplot as plt

# def evaluate():
#     print("evaluate() not defined")

# def test():
#     print("test() not defined")

# def train():
#     print("train() not defined")

network = 'vgg19'  # vgg19 or resnet50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if network == 'vgg19' :  
    model = VGG().to(device)
elif network == 'resnet50' :
    model = ResNet().to(device)
batch_size = 16
learning_rate = 0.001
num_epochs = 50
train_acc = []
valid_acc = []

if __name__ == "__main__" :
    train_data = BufferflyMothLoader("dataset/", "train")
    valid_data = BufferflyMothLoader("dataset/", "valid")
    test_data = BufferflyMothLoader("dataset/", "test")

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay = 0.005, momentum = 0.9)
    
    # Training loop
    best_val = 0
    best_epoch = -1
    for epoch in range(num_epochs) :
        # Train
        model.train()
        correct = 0
        val_correct = 0
        for i, (images, labels) in enumerate(train_loader):
            img = images.to(device)
            gt = labels.to(device)
            outputs = model(img)
            loss = criterion(outputs, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.to(device)).sum().item()
        print('Epoch:', epoch+1, ', Training accuracy:', 100 * round(correct/12594 , 4), '%', end = ' ')
        train_acc.append(100 * round(correct/12594 , 4))
        # Evaluation
        model.eval()
        with torch.no_grad() :  
            for images, labels in valid_loader:
                outputs = model(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels.to(device)).sum().item()
        print(', Valid accuracy:', 100 * val_correct / 500, '%')
        valid_acc.append(100 * val_correct / 500)
        if val_correct > best_val :
            best_val = val_correct
            best_epoch = epoch+1
            if epoch+1 > 30 :
                torch.save(model.state_dict(), './' + network + '/best_model_' + str(epoch+1) + '.pt')

    # Test
    model.eval()
    with torch.no_grad() :
        correct = 0
        for images, labels in test_loader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.to(device)).sum().item()
    print(network + 'Final model Test Accuracy: ',100 * correct / 500,'%')
    torch.save(model.state_dict(), './' + network + '/final_model.pt')

    model.load_state_dict(torch.load('./' + network + '/best_model_' + str(best_epoch) + '.pt'))
    model.eval()
    with torch.no_grad() :
        correct = 0
        for images, labels in test_loader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.to(device)).sum().item()
    print(network + 'Best model Test Accuracy: ',100 * correct / 500,'%')

    # Plot
    idx = []
    for i in range(1, 51) : idx.append(i)
    label_1 = network + '_train_acc'
    label_2 = network + '_valid_acc'
    plt.figure(figsize=(6,5))
    plt.plot(idx, train_acc, label = label_1)    
    plt.plot(idx, valid_acc, label = label_2)
    plt.legend([label_1, label_2])
    plt.xlabel('Epoch')    
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve', loc='left')
    plt.title(network, loc='right')
    plt.savefig('./' + network + '_accuracy_curve.jpg')