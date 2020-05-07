import os
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import random
from torch import randperm
import torch.utils.data as data
import torchvision
import warnings
import skimage.transform
from torch.utils.data import dataset
from PIL import Image


thispath = '/home/jingxiongli/datasets/staindataset'



class Dataset():
    def __init__(self):
        pass

def normalize(sample, maxval):
    """Scales images to be roughly [0,255]."""
    # sample = (2 * (sample.astype(np.float32) / maxval) - 1.) * 1024
    sample = (sample.astype(np.float32) / maxval) * 255
    return sample

def get_targets_abs(path):
    out = []
    targets_path = os.listdir(path)
    for i in targets_path:
        out.append(path + '/' + i)
    return out

class stain_dataset(Dataset):
    def __init__(self,
                 imgpath,
                 transform=None,
                 data_aug=None,
                 seed=0,):

        super(stain_dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug


        # Load data
        self.train_path = imgpath
        self.MAXVAL = 255  # Range [0 255]

        # get data path
        self.train_targets =  get_targets_abs(self.train_path + '/train/' + '4_new') + \
                              get_targets_abs(self.train_path + '/train/' + '5_old') + \
                              get_targets_abs(self.train_path + '/val/' + '4_new') + \
                              get_targets_abs(self.train_path + '/val/' + '5_old')
        # Get our classes.
        self.train_labels = [0]*1000+[1]*1000+ [0]*1000+[1]*1000

        self.train_lables = np.asarray(self.train_labels).T
        self.train_lables = self.train_lables.astype(np.float32)

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, idx):
        img_path = self.train_targets[idx]
        # print(img_path)
        img = Image.open(img_path)
        img = img.resize((224,224),Image.BILINEAR)
        img = np.array(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.data_aug is not None:
            img = self.data_aug(img)

        return {"PA": img, "lab": self.train_labels[idx], "idx": idx}

class stain_dataloader:
    def __init__(self, config, transform, augmentation):
        #  dataset configuration
        self.dataset_path = config.dataset_path
        self.k_fold = config.k_fold
        self.transform = transform
        self.augmentation  = augmentation
        self.train_set, self.val_set = self.init_dataset()
        self.train_iter = IterLoader(data.DataLoader(self.train_set, config.batch_size, shuffle=True, num_workers=2, drop_last=False))
        self.val_iter = IterLoader(data.DataLoader(self.val_set, config.batch_size, shuffle=True, num_workers=2, drop_last=False))


    def init_dataset(self):
        train_dataset_all = stain_dataset(
            imgpath=os.path.join(thispath),
            transform=self.transform,
            data_aug=self.augmentation,
            seed=random.randint(0, 9))

        # val_dataset = stain_dataset(
        #     imgpath=os.path.join(thispath, 'val'),
        #     transform=self.transform,
        #     data_aug=self.augmentation,
        #     seed=random.randint(0, 9)
        # )
        train_test_sets = data.random_split(train_dataset_all, [int(train_dataset_all.__len__() * 1/2),
                                                                                 int(train_dataset_all.__len__()* 1/2)])
        print('train:', len(train_test_sets[0]), 'test:', len(train_test_sets[1]))

        return train_test_sets[0], train_test_sets[1]
        # return train_dataset_all, val_dataset


class IterLoader:
    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(self.loader)

    def next_one(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)

if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision.transforms as t

    # Configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/home/jingxiongli/PycharmProjects/lungDatasets/')
    parser.add_argument('--k_fold', default=True)
    parser.add_argument('--batch_size', type=int, default=32)

    config = parser.parse_args()


    transform = torchvision.transforms.Compose([t.ToPILImage(),
    ])
    aug = torchvision.transforms.RandomApply([t.ColorJitter(brightness=0.5, contrast=0.7),
                                              t.RandomRotation(120),
                                              t.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.75, 1.33),
                                                                  interpolation=2),
                                              t.RandomHorizontalFlip(),
                                              t.RandomVerticalFlip(),
                                              ],p=0.5)
    aug = torchvision.transforms.Compose([aug, t.ToTensor(),])


    stain_dataset = stain_dataloader(config, transform, aug)

    c = stain_dataset.train_iter.next_one()
    print(c['PA'].shape)
    for i, item in enumerate(c['PA']):
        print(item.shape)
        print(c['lab'])
        x = np.moveaxis(np.array(item), 0, -1)
        plt.imshow(x)
        plt.show()
    print('0')