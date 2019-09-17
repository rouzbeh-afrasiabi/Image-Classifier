from utils import *
from model_utils import *
from data_utils import *
import os
import sys
import torch



cwd = os.getcwd()
sys.path.append(cwd)
sys.path.insert(0, cwd)

command_args=get_input()
train_loader,train_dataset=load_and_transform(command_args,'train','random' )
valid_loader,valid_dataset=load_and_transform(command_args,'valid','normal' )
model=Network(command_args)  
train_model(model,command_args, train_loader,valid_loader)
