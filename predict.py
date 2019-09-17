from utils import *
from model_utils import *
from data_utils import *
import os
import sys
import torch


cwd = os.getcwd()
sys.path.append(cwd)
sys.path.insert(0, cwd)

new_command_args=get_input_predict()
checkpoint_files,log_files=search_checkpoint(cwd)
command_args,checkpoint_to_load=select_checkpoint(new_command_args,checkpoint_files,log_files)
model=load_checkpoint(command_args,checkpoint_to_load)
if (model):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and new_command_args.use_gpu) else "cpu")
    print('using '+str(device)+' for inference')
    model=model.to(device)
    train_loader,train_dataset=load_and_transform(command_args,'train','random' )
    output_topc,output_max=predict(model,train_dataset, new_command_args.file_path,new_command_args)