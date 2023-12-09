#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os
import random
from PIL import Image


# In[2]:


torch.__version__


# In[3]:


train_dir = 'C:\\Users\\mahima joshi\\Downloads\\Dogs_Vs_Cats\\Dogs_Vs_Cats\\train'
test_dir = 'C:\\Users\\mahima joshi\\Downloads\\Dogs_Vs_Cats\\Dogs_Vs_Cats\\test'
image_path = 'C:\\Users\\mahima joshi\\Downloads\\Dogs_Vs_Cats\\Dogs_Vs_Cats'


# In[4]:


def walk_through_dir(dir_path):
    for dirpath,dirnames,filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'")

walk_through_dir(image_path)


# In[5]:


from pathlib import Path
random.seed(42)
image_path_list=list(Path(image_path).glob("*/*/*.jpg"))
random_image_path=random.choice(image_path_list)
print(random_image_path)


# In[6]:


image_class=random_image_path.parent.stem
print(image_class)


# In[7]:


img=Image.open(random_image_path)
print(f"Random Image Path:{random_image_path}")
print(f"Image Class:{image_class}")
print(f"Image height:{img.height}")
print(f"Image Width:{img.width}")
img


# In[8]:


import numpy as np
import matplotlib.pyplot as plt
image_as_array = np.asarray(img)
plt.figure(figsize=(10,7))
plt.imshow(image_as_array)
plt.title(f"Image class:{image_class} | Image shape:{image_as_array.shape}")
plt.axis(False)


# In[9]:


from torch.utils.data import DataLoader
from torchvision import datasets,transforms

data_transform =transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

def plot_transformed_images(image_paths: list,transform,n=3,seed=42):
    if seed:
        random.seed(42)
    random_image_paths = random.sample(image_paths,k=n)
    
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig,ax=plt.subplots(nrows=1,ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original\n Size:{f.size}")
            ax[0].axis(False)
            transformed_image=transform(f).permute(1,2,0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \n Shape:{transformed_image.shape}")
            ax[1].axis("off")
            fig.suptitle(f"class :{image_path.parent.stem}",fontsize=16)
            
plot_transformed_images(image_paths=image_path_list,transform=data_transform,n=3,seed=42)
            


# In[10]:


train_data =datasets.ImageFolder( root=train_dir,
                                transform=data_transform,
                                target_transform=None)
test_data=datasets.ImageFolder(root=test_dir,
                              transform=data_transform)
train_data,test_data


# In[11]:


class_names=train_data.classes
class_names


# In[12]:


class_dict=train_data.class_to_idx
class_dict


# In[13]:


from torch.utils.data import Dataset
from typing import Tuple,Dict,List

target_dir =train_dir
print(f"Target dir :{target_dir}")
class_names_found=sorted([entry.name for entry in list(os.scandir(target_dir))])
class_names_found


# In[14]:


list(os.scandir(target_dir))


# In[15]:


def find_classes(directory:str) -> Tuple[List[str],Dict[str,int]]:
    classes=sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    if not classes:
        raise FileNotFoundError(f"Couldn't find the classes in {directory} please check file structure.")
    class_to_idx={class_name: i for i, class_name in enumerate(classes)}
    return classes,class_to_idx

find_classes(target_dir)


# In[16]:


class ImageFolderCustom(Dataset):
    def __init__(self,target_dir:str,transform:None):
        self.paths=list(Path(target_dir).glob("*/*.jpg"))
        self.transform=transform
        self.classes,self.class_to_idx =find_classes(target_dir)
    def load_image(self,index:int)->Image.Image:
        image_path=self.paths[index]
        return Image.open(image_path)
    def __len__(self)->int:
        return len(self.paths)
    def __getitem__(self,index:int)->Tuple[torch.Tensor,int]:
        img=self.load_image(index)
        class_name=self.paths[index].parent.name
        class_idx=self.class_to_idx[class_name]
        if self.transform:
            return self.transform(img),class_idx
        else:
            return img,class_idx


# In[17]:


train_transforms=transforms.Compose([
                    transforms.Resize(size=(64,64)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor()
    
])

test_transforms=transforms.Compose([
                    transforms.Resize(size=(64,64)),
                    transforms.ToTensor
])


# In[18]:


train_data_custom=ImageFolderCustom(target_dir=train_dir,transform=train_transforms)
test_data_custom=ImageFolderCustom(target_dir=test_dir,transform=test_transforms)
train_data_custom,test_data_custom


# In[19]:


len(train_data),len(train_data_custom),len(test_data),len(test_data_custom)
#len(train_data_custom),len(test_data_custom)


# In[20]:


train_data_custom.classes


# In[21]:


train_data_custom.class_to_idx


# In[22]:


def display_random_images(dataset:torch.utils.data.Dataset,classes:List[str]=None,n:int=10,display_shape:bool=True,seed:int=None):
    if n>10:
        n=10
        display_shape=False
        print(f"for display purposes n shouldn't be greater than 10 setting n as 10")
    if seed:
        random.seed(seed)
    random_sample_idx=random.sample(range(len(dataset)),k=n)
    plt.figure(figsize=(16,8))
    for i,target_sample in enumerate(random_sample_idx):
        target_image,target_label =dataset[target_sample][0],dataset[target_sample][1]
        target_image_adjust=target_image.permute(1,2,0)
        plt.subplot(1,n,i+1)
        plt.imshow(target_image_adjust)
        plt.axis("off")
        if classes:
            title=f"class:{classes[target_label]}"
            if display_shape:
                title=title+ f"\nshape;{target_image_adjust.shape}"
        plt.title(title)

display_random_images(train_data,n=5,classes=class_names,seed=None)
            


# In[23]:


display_random_images(train_data_custom,n=20,classes=class_names,seed=None)


# In[24]:


BATCH_SIZE=32
NUM_WORKERS=os.cpu_count()
train_dataloader_custom=DataLoader(dataset=train_data_custom,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,shuffle=True)
test_dataloader_custom=DataLoader(dataset=test_data_custom,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,shuffle=False)
train_dataloader_custom,test_dataloader_custom


# In[ ]:


# from tqdm import tqdm
# img_custom,label_custom=tqdm(next(iter(train_dataloader_custom)))
# img_custom.shape,label_custom.shape


# In[25]:


train_transform=transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

test_transform=transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor()
])


# In[26]:


image_path_list=list(Path(image_path).glob("*/*/*.jpg"))
image_path_list[:10]


# In[27]:


simple_transform=transforms.Compose([transforms.Resize(size=(64,64)),
                                    transforms.ToTensor()
                                    ])


# In[28]:


train_data_simple=datasets.ImageFolder(
                    root=train_dir,
                    transform=simple_transform)
test_data_simple=datasets.ImageFolder(
                    root=test_dir,
                    transform=simple_transform)


# In[29]:


BATCH_size=32
NUM_WORKERS=os.cpu_count()

train_dataloader_simple=DataLoader(dataset=train_data_simple,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS)

test_dataloader_simple=DataLoader(dataset=test_data_simple,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False,
                                  num_workers=NUM_WORKERS)


# In[30]:


from torch import nn
class TinyVGG(nn.Module):
    """Model architecture copying Tiny VGG
    from CNN Explainer :poloclub.github.io"""
    
    def __init__(self,input_shape:int,hidden_units:int,output_shape:int)-> None:
        super().__init__()
        self.conv_block_1=nn.Sequential(
                nn.Conv2d(in_channels=input_shape,
                       out_channels=hidden_units,
                       kernel_size=3,
                       stride=1,
                       padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                       out_channels=hidden_units,
                       kernel_size=3,
                       stride=1,
                       padding=0),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2,
                        stride=1)
            )
        self.conv_block_2=nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                     out_channels=hidden_units,
                     kernel_size=3,
                     stride=1,
                     padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                     kernel_size=3,
                      stride=1,
                     padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                        stride=2)
            )
            
        self.classifier=nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(in_features=hidden_units*27*27,
                                     out_features=output_shape)
                            )
    def forward(self,x):
        x=self.conv_block_1(x)
        #print(x.shape)
        x=self.conv_block_2(x)
        #print(x.shape)
        x=self.classifier(x)
        #print(x.shape)
        return x
        


# In[31]:


torch.manual_seed(42)
model_0=TinyVGG(input_shape=3,
               hidden_units=10,
               output_shape=len(class_names))
model_0


# In[32]:


image_batch,label_batch=next(iter(train_dataloader_simple))
image_batch.shape,label_batch.shape


# In[33]:


model_0(image_batch)
# train_dataloader_simple


# In[34]:


import torchinfo
from torchinfo import summary
summary(model_0,input_size=[1,3,64,64])


# In[35]:


def train_step(model:torch.nn.Module,
              dataloader:torch.utils.data.DataLoader,
              loss_fn:torch.nn.Module,
              optimizer:torch.optim.Optimizer,
              device=None):
    model.train()
    train_loss,train_acc=0,0
    for batch,(X,y) in enumerate(dataloader):
        y_pred=model(X)
        
        loss=loss_fn(y_pred,y)
        train_loss+=loss.item()
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        y_pred_class =torch.argmax(
        torch.softmax(y_pred,dim=1),dim=1)
        train_acc+=(y_pred_class==y).sum().item()/len(y_pred)
        
    train_loss=train_loss/len(dataloader)
    train_acc=train_acc/len(dataloader)
        
    return train_loss,train_acc    


# In[36]:


def test_step(model:torch.nn.Module,
             dataloader:torch.utils.data.DataLoader,
             loss_fn:torch.nn.Module,
             device=None):
    model.eval()
    test_loss,test_acc=0,0
    
    with torch.inference_mode():
        for batch,(X,y) in enumerate(dataloader):
            test_pred_logits=model(X)
            
            loss=loss_fn(test_pred_logits,y)
            test_loss+=loss.item()
            
            test_pred_labels=test_pred_logits.argmax(dim=1)
            
            test_acc+=((test_pred_labels==y).sum().item()/len(test_pred_labels))
    test_loss=test_loss/len(dataloader)
    test_acc=test_acc/len(dataloader)
    return test_loss,test_acc


# In[37]:


from tqdm.auto import tqdm
def train(model:torch.nn.Module,
         train_dataloader:torch.utils.data.dataloader,
         test_dataloader:torch.utils.data.dataloader,
         optimizer:torch.optim.Optimizer,
         loss_fn:torch.nn.Module=nn.CrossEntropyLoss(),
         epochs:int=5,
         device=None):
    results={"train_loss":[],
            "train_acc":[],
            "test_loss":[],
            "test_acc":[]}
    
    for epoch in tqdm(range(epochs)):
        train_loss,train_acc=train_step(model=model,
                                       dataloader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       device=None)
        test_loss,test_acc=test_step(model=model,
                                    dataloader=test_dataloader,
                                    loss_fn=loss_fn,
                                    device=None)
        
        print(f"Epoch:{epoch} | Train Loss: {train_loss:.4f} | Train accuracy: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
    return results
    
    

            


# In[38]:


torch.manual_seed(42)
NUM_EPOCHS=5
model_0 = TinyVGG(input_shape=3,
                 hidden_units=10,
                 output_shape=len(train_data.classes))


# In[39]:


loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(params=model_0.parameters(),lr=0.001)


# In[40]:


from timeit import default_timer as timer
start_timer=timer()


# In[41]:


model_0_results= train(model=model_0,
                      train_dataloader=train_dataloader_simple,
                      test_dataloader=test_dataloader_simple,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      epochs=NUM_EPOCHS)
end_timer=timer()
print(f"Total training time :{end_timer-start_timer:.3f} seconds")

model_0_results


# In[42]:


model_0_results.keys()


# In[43]:


def plot_loss_curves(results:Dict[str,list[float]]):
    loss=results["train_loss"]
    test_loss=results["test_loss"]
    
    accuracy=results["train_acc"]
    test_accuracy=results["test_acc"]
    
    epoches=range(len(results["train_loss"]))
    
    plt.figure(figsize=(15,7))
    
    plt.subplot(1,2,1)
    plt.plot(epoches,loss,label="train_loss")
    plt.plot(epoches,test_loss,label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(epoches,accuracy,label="train_accuracy")
    plt.plot(epoches,test_accuracy,label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend();
    


# In[44]:


plot_loss_curves(model_0_results)


# In[45]:


import PIL
import torchvision.transforms.functional as transform

# Reads a file using pillow
PIL_image = PIL.Image.open("C:\\Users\\mahima joshi\\Desktop\\check\\pic1.jpg")

# The image can be converted to tensor using
tensor_image = transform.to_tensor(PIL_image)


# In[46]:


plt.imshow(tensor_image.permute(1,2,0))


# In[47]:


print(f"custom image tensor: {tensor_image}")
print(f"custom image shape: {tensor_image.shape}")
print(f"custom image datatype: {tensor_image.dtype}")


# In[55]:





# In[58]:





# In[52]:


custom_image_transform=transforms.Compose([
    transforms.Resize(size=(64,64))
])


# In[56]:


custom_image_transformed = custom_image_transform(tensor_image)
print(f"Original shape: {tensor_image.shape}")
print(f"Transformed shape: {custom_image_transformed.shape}")


# In[57]:


plt.imshow(custom_image_transformed.permute(1,2,0))


# In[ ]:





# In[62]:


custom_image_transformed.shape,custom_image_transformed.unsqueeze(0).shap


# In[64]:


model_0.eval()
with torch.inference_mode():
    custom_image_pred =model_0(custom_image_transformed.unsqueeze(0))
custom_image_pred


# In[65]:


class_names


# In[66]:


custom_image_pred_probs=torch.softmax(custom_image_pred,dim=1)
custom_image_pred_probs


# In[68]:


custom_image_pred_labels=torch.argmax(custom_image_pred_probs,dim=1)
custom_image_pred_labels


# In[81]:


import PIL
import torchvision.transforms.functional as transform

def pred_plot_img(model:torch.nn.Module,
                 image_path:str,
                 class_names:List[str]=None,
                 device=None):
    # Reads a file using pillow
    
    PIL_image = PIL.Image.open(image_path)

    # The image can be converted to tensor using
    tensor_image = transform.to_tensor(PIL_image)
    
    custom_image_transformed = custom_image_transform(tensor_image)
    
    model.eval()
    
    with torch.inference_mode():
        custom_image_pred =model_0(custom_image_transformed.unsqueeze(0))
        
    target_pred=torch.softmax(custom_image_pred,dim=1)
    
    target_image_pred_labels=torch.argmax(target_pred,dim=1)
    
    plt.imshow(tensor_image.permute(1,2,0))
    
    if class_names:
        title= f"Pred:{class_names[target_image_pred_labels]} | Prob: {target_pred.max():.3f}"
    else:
        title= f"Pred:{target_image_pred_labels} | Prob: {target_pred.max():.3f}"
    plt.title(title)
    plt.axis(False)


# In[83]:


pred_plot_img(model=model_0,
             image_path="C:\\Users\\mahima joshi\\Desktop\\check\\pic2.jpg",
             class_names=class_names,
             device=None)


# In[ ]:




