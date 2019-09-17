from torchvision import datasets, transforms
import torch
import os
from torchvision.transforms import ToPILImage
import torchvision
from torchvision.utils import make_grid
from PIL import Image
import numpy as np



cwd = os.getcwd()
show_image=ToPILImage()

def show_grid(loader):    
    def image_grid(input_images):
        input_images = np.array([0.229, 0.224, 0.225]) * input_images.numpy().transpose((1, 2, 0)) + np.array([0.485, 0.456, 0.406])
        plt.imshow(np.clip(input_images, 0, 1))

    images, labels = next(iter(loader))
    image_grid(make_grid(images))
    
def load_and_transform (command_args,folder,transform_type='normal'):
    randomized_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.RandomCrop((224,224)),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomGrayscale(p=0.4),
                                      transforms.RandomRotation((0,360)),
                                      #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                                      transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    

    normal_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    if (transform_type=='normal'):
        transform=normal_transform
    elif(transform_type=='random'):
        transform=randomized_transform
       
    data_set=datasets.ImageFolder(os.path.join(cwd,command_args.data_dir+'/'+ folder) , transform)
    loader = torch.utils.data.DataLoader(data_set, batch_size=command_args.batch_size, shuffle=command_args.shuffle)
    return(loader,data_set)