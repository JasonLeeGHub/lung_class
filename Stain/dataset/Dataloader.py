import os
import numpy as np
import random
from torch import randperm
import torch.utils.data as data
import torchvision

from dataset.covid19 import NIH_Dataset, COVID19_Dataset, Merge_Dataset, FilterDataset, relabel_dataset, XRayCenterCrop, XRayResizer, histeq, ToPILImage, Tofloat32

thispath = '/home/jingxiongli/PycharmProjects/lungDatasets/'

class dataset_loader:
    def __init__(self, config, transform, augmentation):
        #  dataset configuration
        self.dataset_path = config.dataset_path
        self.k_fold = config.k_fold
        self.transform = transform
        self.augmentation  = augmentation
        # init loaders
        lung_datasets, train_dataset, test_dataset, dataset_all = self.init_loaders()
        self.train_iter_1 = IterLoader(data.DataLoader(lung_datasets[0][1], config.batch_size, shuffle=True, num_workers=2, drop_last=False))
        self.val_iter_1 = IterLoader(data.DataLoader(lung_datasets[0][0], config.batch_size, shuffle=True, num_workers=2, drop_last=False))
        self.train_iter_2 = IterLoader(data.DataLoader(lung_datasets[1][1], config.batch_size, shuffle=True, num_workers=2, drop_last=False))
        self.val_iter_2 = IterLoader(data.DataLoader(lung_datasets[1][0], config.batch_size, shuffle=True, num_workers=2, drop_last=False))
        self.train_iter_3 = IterLoader(data.DataLoader(lung_datasets[2][1], config.batch_size, shuffle=True, num_workers=2, drop_last=False))
        self.val_iter_3 = IterLoader(data.DataLoader(lung_datasets[2][0], config.batch_size, shuffle=True, num_workers=2, drop_last=False))
        self.train_iter_4 = IterLoader(data.DataLoader(lung_datasets[3][1], config.batch_size, shuffle=True, num_workers=2, drop_last=False))
        self.val_iter_4 = IterLoader(data.DataLoader(lung_datasets[3][0], config.batch_size, shuffle=True, num_workers=2, drop_last=False))
        self.train_iter_5 = IterLoader(data.DataLoader(lung_datasets[4][1], config.batch_size, shuffle=True, num_workers=2, drop_last=False))
        self.val_iter_5 = IterLoader(data.DataLoader(lung_datasets[4][0], config.batch_size, shuffle=True, num_workers=2, drop_last=False))

        self.train_iter = IterLoader(data.DataLoader(train_dataset, config.batch_size, shuffle=True, num_workers=2, drop_last=False))
        self.test_iter = IterLoader(data.DataLoader(test_dataset, config.batch_size, shuffle=True, num_workers=2, drop_last=False))
        self.all_iter = IterLoader(data.DataLoader(dataset_all, config.batch_size, shuffle=True, num_workers=2, drop_last=True))


    def init_loaders(self):
        test_covid_dataset = COVID19_Dataset(
                 imgpath=os.path.join(thispath, "covid-chestxray-dataset", "images"),
                 csvpath=os.path.join(thispath, "covid-chestxray-dataset", "metadata.csv"),
                 views=["PA"],
                 transform=self.transform,
                 data_aug=self.augmentation,
                 nrows=None,
                 seed=random.randint(0,9))
        filtered_covid = FilterDataset(test_covid_dataset, labels=["COVID-19"])
        # relabel
        relabel_dataset(pathologies=["COVID-19", "Pneumonia"], dataset=filtered_covid)
        i, _ = list(enumerate(filtered_covid.labels))[-1]
        # relabel = np.array([1.0, 0.0])
        # filtered_covid.labels = np.expand_dims(relabel, 0).repeat(i + 1, axis=0)
        relabel = np.array([0.0])
        filtered_covid.labels = relabel.repeat(i + 1, axis=0)

        test_NIH_dataset = NIH_Dataset(
                 imgpath = os.path.join(self.dataset_path, "NIH", "images-224"),
                 csvpath=os.path.join(self.dataset_path, "NIH", "Data_Entry_2017.csv"),
                 transform=self.transform,
                 data_aug=self.augmentation,
                 nrows=None,
                 seed=random.randint(0,9),
                 pure_labels=False,
                 unique_patients=True)
        filtered_NIH = FilterDataset(test_NIH_dataset, labels=['Pneumonia'])
        relabel_dataset(pathologies=["COVID-19", "Pneumonia"], dataset=filtered_NIH)
        i, _ = list(enumerate(filtered_NIH.labels))[-1]
        # relabel = np.array([0.0, 1.0])
        # filtered_NIH.labels = np.expand_dims(relabel, 0).repeat(i + 1, axis=0)
        relabel = np.array([1.0])
        filtered_NIH.labels = relabel.repeat(i + 1, axis=0)

        dataset_train_all = Merge_Dataset((filtered_covid, filtered_NIH), seed=1)
        train_eval_sets = []
        if self.k_fold:
            # spilt dataset (random split)
            train_test_sets = data.random_split(dataset_train_all, [int(dataset_train_all.length * 5/ 6),
                                                                         int(dataset_train_all.length* 1/6)])
            print('train:', len(train_test_sets[0]), 'test:', len(train_test_sets[1]))

            # spilt dataset (k fold)
            k = 5
            val_lenth = int(len(train_test_sets[0]) * 1 / k)
            train_lenth = int(len(train_test_sets[0]) * (k-1) / k)
            lengths = [val_lenth,train_lenth]
            indices_base = randperm(val_lenth+train_lenth).tolist()

            for i in range(5):
                dataset_cache = [data.Subset(dataset_train_all, indices_base[(offset - length):offset]) for offset, length in zip(_accumulate(lengths), lengths)]
                indices_base = np.concatenate([indices_base[val_lenth:-1],indices_base[0 : val_lenth]])
                train_eval_sets.append(dataset_cache)


        return train_eval_sets, train_test_sets[0],train_test_sets[1], dataset_train_all

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

# pneumonias = ["COVID-19", "SARS", "MERS", "ARDS", "Streptococcus", "Pneumocystis", "Klebsiella",
#               "Chlamydophila", "Legionella"]
# mapping["Pneumonia"] = pneumonias
# mapping["Viral Pneumonia"] = ["COVID-19", "SARS", "MERS"]
# mapping["Bacterial Pneumonia"] = ["Streptococcus", "Klebsiella", "Chlamydophila", "Legionella"]
# mapping["Fungal Pneumonia"] = ["Pneumocystis"]

def _accumulate(iterable, fn=lambda x, y: x + y):
    'Return running totals'
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total



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
    transform = torchvision.transforms.Compose([XRayCenterCrop(),
                                                XRayResizer(256),
                                                ToPILImage(),
                                                t.CenterCrop(200),
                                                t.Grayscale(num_output_channels=3),
                                                t.Resize(224),
                                                ])

    aug = torchvision.transforms.RandomApply([t.ColorJitter(brightness=0.5, contrast=0.7),
                                              t.RandomRotation(120),
                                              t.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.75, 1.33), interpolation=2),
                                              t.RandomHorizontalFlip(),
                                              t.RandomVerticalFlip(),
                                          ], p=0.5)
    # random_hist = torchvision.transforms.Compose([histeq()])

    aug = torchvision.transforms.Compose([t.ToTensor(),Tofloat32()])
    # aug = None


    lung_dataset = dataset_loader(config, transform, aug)
    print('x')
    c = lung_dataset.test_iter.next_one()
    from PIL import Image
    from pylab import array,flatten, hist, show
    for i, item in enumerate(c['PA']):
        x = histeq()
        item_histeq = x(item)
        show()
        plt.subplot(2,2,1)
        plt.imshow(np.squeeze(item[0]), cmap='gray')
        plt.axis('off')
        plt.subplot(2,2,2)
        plt.hist(x=item[0].flatten()*255,bins=64, histtype='bar', range=(0,255), density=False)
        plt.ylim((0,4500))
        plt.xlabel('Pixel Vlaue')
        plt.ylabel('Counts')
        # plt.axis('off')

        plt.subplot(2,2,3)
        plt.imshow(np.squeeze(item_histeq[0]), cmap='gray')
        plt.axis('off')
        plt.subplot(2,2,4)
        plt.hist(x=item_histeq[0].flatten(),bins=64, histtype='bar', range=(0,255))
        plt.ylim((0,4500))
        plt.xlabel('Pixel Vlaue')
        plt.ylabel('Counts')
        # plt.axis('off')
        plt.show()
    print('0')