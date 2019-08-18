from __future__ import print_function
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
#from .utils import noisify

#root="/home/caozhantao/Co-teaching/Co-teaching/"

# -----------------ready the dataset--------------------------
def default_loader(path):
    im = Image.open(path).convert('RGB')
    out = im.resize((227, 227))
    #out = im.resize((32, 32))
    #out = im.resize((180, 180))
    return out
    #.convert('RGB')


class MEDICAL(data.Dataset):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False,
		        noise_type=None, noise_rate=0.2, random_state=0, loader=default_loader, aug = None):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        #print (self.target_transform)
        self.train = train  
        # training set or test set
        self.dataset='medical'
        self.noise_type=noise_type
        self.loader = loader
        self.aug = aug

        if self.train:
            self.train_data, self.train_labels, self.real_train_labels = self.ReadImage(self.root, "train/", "train.txt")

            print (len(self.train_data))
            print (self.train_labels)

            self.train_labels=np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
            _train_labels=[i[0] for i in self.train_labels]
            self.noise_or_not = np.transpose(_train_labels)==np.transpose(_train_labels)
            if noise_type != 'clean':
                pass
        else:
            self.test_data, self.test_labels, self.real_test_labels = self.ReadImage(self.root, "val/", "val.txt")
            print (len(self.test_data))

    def ReadImage(self, root, middle, file_name):
        fh = open(root + middle + file_name, 'r')
        datas = []
        labels = []
        real_labels = []

        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            image_path = root + middle + words[0]
            img = self.loader(image_path)
            #print (int(words[1]))
            #print (image_path)
            datas.append(img)
            labels.append(int(words[1]))
            if (int(os.path.splitext(words[0])[0]) > 1000000):
                #print (os.path.splitext(words[0])[0])
                real_labels.append(1)
            else:
                real_labels.append(0)

            #print (image_path, middle,int(words[1]))
        #self.imgs = imgs
        return datas, labels, real_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target, clean = self.train_data[index], self.train_labels[index], self.real_train_labels[index]
        else:
            img, target, clean = self.test_data[index], self.test_labels[index], self.real_test_labels[index]

        if self.transform is not None:
            if self.aug is not None:
                
                img_array = np.asarray(img)
                #print ("source",img_array)
                student = self.aug.augment(img_array)
                #print ("target",student)
              
                img = Image.fromarray(np.uint8(student))

                            
                
                
            
            img = self.transform(img)

        #print (self.target_transform)
        if self.target_transform is not None:
            target = self.target_transform(target)
            #print ("=========================")

        #print ("img", img.size())
        #print ("target",target)
        return img, target, index, clean

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

class MEDICALaaa(data.Dataset):
    def __init__(self, root, middle, file_name, transform=None, target_transform=None, loader=default_loader):
        fh = open(root + middle+ file_name, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            image_path = root + middle + words[0]
            imgs.append((image_path,int(words[1])))
            #print (image_path, int(words[1]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, index

    #def __len__(self):
    #    return len(self.imgs)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

#train_data= MEDICAL(root, True, transform=transforms.ToTensor())
#test_data= MEDICAL(root, True, transform=transforms.ToTensor())
#train_loader = data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
#test_loader = data.DataLoader(dataset=test_data, batch_size=64)