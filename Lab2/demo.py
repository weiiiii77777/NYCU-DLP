import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from dataloader import BufferflyMothLoader
from VGG19 import VGG
from ResNet50 import ResNet

backbone = 'resnet50'  # vgg19 or resnet50
type = 'valid'
model_name = 'best'
num = 918
fast = 1 # 0 or 1

device = torch.device('cuda:2')
batch_size = 16
test_data = BufferflyMothLoader("dataset/", "test")
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

if not fast :
    valid_data = BufferflyMothLoader("dataset/", "valid")
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)

if fast :
    model_1 = VGG().to(device)
    model_1.load_state_dict(torch.load('./vgg19/model_best.pt'))
    model_1.eval()
    correct = 0
    with torch.no_grad() :
        for images, labels in test_loader:
            outputs = model_1(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.to(device)).sum().item()
    print('VGG19 Test Data Accuracy : ',100 * correct / 500,'%')

    model_2 = ResNet().to(device)
    model_2.load_state_dict(torch.load('./resnet50/model_best.pt'))
    model_2.eval()
    correct = 0
    with torch.no_grad() :
        for images, labels in test_loader:
            outputs = model_2(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.to(device)).sum().item()
    print('ResNet50 Test Data Accuracy : ',100 * correct / 500,'%')


else :

    if backbone == 'vgg19' : 
        model = VGG().to(device)
    elif backbone == 'resnet50' : 
        model = ResNet().to(device)

    if num :
        # model.load_state_dict(torch.load('./'+backbone+'/'+model_name+'_model_'+str(num)+'.pt'))
        model.load_state_dict(torch.load('./'+backbone+'/model_'+str(num)+'.pt'))
    else :
        model.load_state_dict(torch.load('./'+backbone+'/'+model_name+'_model.pt'))

    model.eval()
    with torch.no_grad() :
        correct = 0
        if type == 'valid' :
            for images, labels in valid_loader:
                outputs = model(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels.to(device)).sum().item()
        elif type == 'test' :
            for images, labels in test_loader:
                outputs = model(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels.to(device)).sum().item()
    print(model_name+' model '+type+' Accuracy: ',100 * correct / 500,'%')
