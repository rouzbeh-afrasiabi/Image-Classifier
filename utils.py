import argparse
import os
import sys
import torch
import json
from datetime import datetime
import numpy as np

cwd = os.getcwd()
current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict) 
    
def check_file(filename,folder):    
    exists=False
    if(check_folder(folder)):   
        for root, dirs, files in os.walk(folder):
            if(filename in  [file for file in files]):
                exists=True
    else:
        exists=False
    return exists

def check_folder(foldername):    
    exists=False
    for root, dirs, files in os.walk(cwd):
        if(foldername in  [dir for dir in dirs]):
            exists=True
    while(not exists):
        print('folder '+foldername,' does not exist in '+cwd)
        user_input = input("Create folder? (Y/N)")
        if(user_input =="Y"):
            os.mkdir(cwd+'/'+foldername)
            exists=True
            break
        else:
            print('Using default folder ','saved_data')
            if(not check_folder('saved_data')):
                os.mkdir(cwd+'/saved_data')
                exists=True
                break
    return exists

def search_checkpoint(folder):
    checkpoint_files=[]
    log_files=[]
    for root, dirs, files in os.walk(folder):
        for dir in dirs:
            for child_root, child_dirs, child_files in os.walk(dir):
                for filename in child_files:
                    if('.pth' in filename):
                        if (os.path.join(folder,dir, filename) not in checkpoint_files):
                            checkpoint_files.append(os.path.join(folder,dir, filename))
                    if('.log' in filename):
                        if (os.path.join(folder,dir, filename) not in checkpoint_files):
                            log_files.append(os.path.join(folder,dir, filename))
    return(checkpoint_files,log_files)
                    
def select_checkpoint(new_command_args,checkpoint_files,log_files):
    user_input=0
    checkpoint_to_load=None
    if(not new_command_args.checkpoint):
        print('following checkpoint files were found, please select one to continue\n')
        for i,x in enumerate(checkpoint_files):
            print([i+1], os.path.basename(x))
        while(user_input==0):
            try:
                user_input = int(input("Select checkpoint to load"+str([1,len(checkpoint_files)])+':'))
            except ValueError:
                print('Value not integer')
                continue
            if(user_input<=len(checkpoint_files) and user_input>=1):
                checkpoint_to_load=(checkpoint_files[user_input-1])
                break
            else:
                user_input=0
                continue
    else:
        if(any(os.path.basename(new_command_args.checkpoint) in x for x in checkpoint_files)):
            index=int(np.where(os.path.basename(new_command_args.checkpoint) in x for x in checkpoint_files)[0])
            checkpoint_to_load=checkpoint_files[index]
    log_found=False
    print(checkpoint_to_load)
    for i,x in enumerate(log_files):
        with open(x,"r") as F:
            for k,line in enumerate(F):
                if (os.path.basename(checkpoint_to_load) in line ):
                    log_found=True
                    timestamp=os.path.basename(checkpoint_to_load).split('.')[-2]
                    specs=json.loads(line)
                    command_args=Bunch(specs[timestamp])
        if(not log_found):
            print('could not find the checkpoint log associated with this checkpoint')
            command_args=''
        return(command_args,checkpoint_to_load)
           
def get_input():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    command_parser = argparse.ArgumentParser(description='Here you can train a model for flower type catagorization')
    command_parser.add_argument('data_dir', action="store", nargs='*', default="/home/workspace/aipnd-project/flowers/")
    command_parser.add_argument('--arch', action="store", dest="model_arch", default="resnet")
    command_parser.add_argument('--save_dir', action="store", dest="save_dir", default='saved_data')
    command_parser.add_argument('--file_name', action="store", dest="file_name", default="checkpoint")
    command_parser.add_argument('--learning_rate', action="store", dest="learning_rate", default=0.001)
    command_parser.add_argument('--momentum', action="store", dest="momentum", default=0.9)
    command_parser.add_argument('--epochs', action="store", dest="epochs", type=int, default=1)
    command_parser.add_argument('--hidden_units', action="store",dest="hidden_units", type=int, default=1020)
    command_parser.add_argument('--gpu',action='store_true',dest="use_gpu",default=False)
    command_parser.add_argument('--batch', action="store", dest="batch_size", type=int, default=64)
    command_parser.add_argument('--shuffle_off',action='store_false',dest="shuffle",default=True)   
    command_parser.add_argument('--print_every',action='store',dest="print_every",default=10)
    
    command_args = command_parser.parse_args()  
    if (command_args.file_name!=""):
        command_args.file_name=command_args.file_name+"."+command_args.model_arch+'.'+current_time+".pth"
    while(command_args.use_gpu and 'cpu' in str(device)):
        print('Warning: GPU is not available, switching to CPU. Training will take much longer using CPU\n')
        user_input = input("Continue? (Y/N)")
        if(user_input=='Y'):
            break
        if(user_input=='N'):
            print('Please turn GPU on and try again')
            user_input = input("Try again? (Y/N)")
            if (user_input=='Y'):
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                continue
            else:
                print('Using CPU then')
                command_args.use_gpu=False
                break
        else:
            print("Use Y for yes and N for No, try again")
            continue
        command_args.use_gpu=False
    while (check_file(command_args.file_name,command_args.save_dir)):
        print('Warning: This will overwrite '+command_args.file_name)
        user_input = input("Continue? (Y/N)")
        if(user_input=='Y'):
            break
        if (user_input=='N'):
            new_name = input("Please provide new name: ")
            if(new_name!=command_args.file_name and new_name!='' and not check_file(new_name,command_args.save_dir)):
                command_args.file_name=new_name
                break
            else:
                continue
            
        else:
            print("Use Y for yes and N for No, try again")
            continue
    log_file=os.path.join(cwd, command_args.save_dir+"/input_log.log")
    with open(log_file, 'a') as f:
        if(os.stat(log_file).st_size != 0):
            f.write('\n'+json.dumps({current_time: vars(command_args)}))
        else:
            f.write(json.dumps({command_args.file_name: vars(command_args)}))
    return(command_args)

def get_input_predict():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    command_parser = argparse.ArgumentParser(description='Here you can perform inference based on a pre-trained model')
    command_parser.add_argument('file_path', action="store", default="")
    command_parser.add_argument('--checkpoint', action="store", default="")
    command_parser.add_argument('--gpu',action='store_true',dest="use_gpu",default=False)
    command_parser.add_argument('--top_k', action="store", dest="top_k", type=int, default=3)
    command_parser.add_argument('--category_names', action="store", dest="category_names", default='cat_to_name.json')    
    
    command_args = command_parser.parse_args()  
    while(command_args.use_gpu and 'cpu' in str(device)):
        print('Warning: GPU is not available, switching to CPU. Training will take much longer using CPU\n')
        user_input = input("Continue? (Y/N)")
        if(user_input=='Y'):
            break
        if(user_input=='N'):
            print('Please turn GPU on and try again')
            user_input = input("Try again? (Y/N)")
            if (user_input=='Y'):
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                continue
            else:
                print('Using CPU then')
                command_args.use_gpu=False
                break
        else:
            print("Use Y for yes and N for No, try again")
            continue
        command_args.use_gpu=False  
    return(command_args)