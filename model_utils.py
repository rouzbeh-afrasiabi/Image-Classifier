import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import json
import re

import torch
from torch import nn
from torch import optim
import torchvision
from torchvision.utils import make_grid
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.transforms import ToPILImage
from PIL import Image
from torch.autograd import Variable
from collections import OrderedDict
from workspace_utils import active_session
import os
import sys
from datetime import datetime

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.insert(0, cwd)

class Network(nn.Module):
    def __init__(self,command_args):
        super().__init__()
        device = torch.device("cuda:0" if (torch.cuda.is_available() and command_args.use_gpu) else "cpu")
        print("loading "+command_args.model_arch+" model\n")
        if (command_args.model_arch=='resnet'):
            self.pre_trained_model = models.resnet152(pretrained=True).to(device)
        elif(command_args.model_arch=='densenet'):
            self.pre_trained_model = models.densenet201(pretrained=True).to(device)
            
        in_features=list(self.pre_trained_model.children())[-1].in_features
        for param in self.pre_trained_model.parameters():
            param.requires_grad = False
        if(command_args.model_arch=='resnet'):
            self.pre_trained_model.fc=nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features, command_args.hidden_units)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(command_args.hidden_units, 510)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(510, 102)),
            ('output', nn.LogSoftmax(dim=1))
                          ]))  
        if(command_args.model_arch=='densenet'):
            self.pre_trained_model.classifier=nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features, command_args.hidden_units)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(command_args.hidden_units, 510)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(510, 102)),
            ('output', nn.LogSoftmax(dim=1))
                          ]))  
    def forward(self,x):
        x=self.pre_trained_model(x)
        return(x)
    
def validation(model, testloader, criterion,device):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        test_loss += criterion(output, labels)

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return (test_loss, accuracy)   
    
def train_model(model,command_args,train_loader,valid_loader,print_every=10):
    with active_session():

        device = torch.device("cuda:0" if (torch.cuda.is_available() and command_args.use_gpu) else "cpu")
        model=model.to(device)
        print('\ntraining in progress ...'+'\nusing '+str(device)+'\n')
        epochs=command_args.epochs
        steps = 0
        print_every=command_args.print_every

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                              lr=command_args.learning_rate,momentum=command_args.momentum)
        test_loss_all=[]
        accuracy_all=[]
        for e in range(epochs):
            running_loss=0
            test_loss_all=[]
            accuracy_all=[]
            for images,labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                steps+=1
                optimizer.zero_grad()
                output=model(images)
                loss=criterion(output,labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                #print('epoch',e,' and ',step,' step ','is running')
                if steps % print_every == 0:
                    model.eval()
                    with torch.no_grad():
                        test_loss, accuracy = validation(model, valid_loader, criterion,device)
                        test_loss_all.append(test_loss)
                        accuracy_all.append(accuracy)
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Test Loss: {:.3f}.. ".format(test_loss/len(valid_loader)),
                          "Test Accuracy: {:.3f}".format(accuracy/len(valid_loader)),
                         )
                    running_loss = 0
                    model.train()
        save_location=os.path.join(cwd, command_args.save_dir+'/'+command_args.file_name)
        checkpoint = {'input_size': list(list(model.children())[1].children())[-1][0].in_features,
                  'output_size': list(list(model.children())[1].children())[-1][-2].out_features,
                  'state_dict': model.state_dict(),
                  'class_to_idx': trainset.class_to_idx,
                  'cat_to_name':cat_to_name,
                  'epochs':epochs,
                  'optimizer':optimizer.state_dict(),
                  'criterion':criterion,
                  'model':model
                 }
        torch.save(model.state_dict(),save_location )
def load_checkpoint(command_args,path):
    if(command_args!='' and os.path.isfile(path)):
        checkpoint = torch.load(path,map_location='cpu')
        model=Network(command_args)
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except:
            model.load_state_dict(checkpoint)
        return(model)
    else:
        if (not os.path.isfile(path)):
            print('invalid file location')
            return False
        if(command_args==''):
            print('checkpoint data is missing from the log file')
            return False
            
def process_image(image):
    normal_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
    try:
        input_img = Image.open(image)
        output_img=normal_transform(input_img)
    except:
        if(len(image)<1):
            print('path to image file not provided')
        else:
            print('\nInvalid path, could not locate : ',image,'\nCurrent working directory is ',cwd )
        output_img=False
    return(output_img)

def predict(model,trainset, image_path,new_command_args):
    top_k=new_command_args.top_k
    index_to_cat=torch.zeros(len(trainset.class_to_idx))
    model.eval()
    processed_image=process_image(image_path)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and new_command_args.use_gpu) else "cpu")
    try:
        processed_image.shape
        processed_image=processed_image.to(device)
        processed_image.unsqueeze_(0)
        with torch.no_grad():
                output = model(processed_image)
                output_exp=torch.exp(output)
                output_max=output_exp.max(dim=1)
                output_topk=torch.topk(output_exp,top_k)

        for i,k in enumerate(trainset.class_to_idx):
            index_to_cat[i]=int(k)
        output_topc=[]
        with open(new_command_args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        for k,i in enumerate([i for i in output_topk[1][0].cpu().numpy()]):
            output_topc.append([cat_to_name[str(int(index_to_cat[i]))],output_topk[0][0].cpu().numpy()[k]])
        for j,x in enumerate(output_topc):
            print([j+1],x)
        return  (output_topc,output_max)
    except:
        print('prediction stopped')
        return(False,False)