import os
import numpy as np
import random
import torch.utils.data as data
import torchvision

from dataset.covid19 import NIH_Dataset, COVID19_Dataset, Merge_Dataset, FilterDataset, relabel_dataset, XRayCenterCrop, XRayResizer, histeq, ToPILImage

thispath = '/home/jingxiongli/PycharmProjects/lungDatasets/'

class dataset_loader:
    def __init__(self, config, transform, augmentation):
        #  dataset configuration
        self.dataset_path = config.dataset_path
        # transforms and augmentation
        self.transform = transform
        self.augmentation  = augmentation
        # init loaders
        self.train_set, self.val_set = self.init_loaders()
        # init iters
        self.train_set_loader = data.DataLoader(self.train_set, 24, shuffle=True, num_workers=8, drop_last=False)
        self.val_set_loader = data.DataLoader(self.val_set, 24, shuffle=True, num_workers=8, drop_last=False)

        self.train_set_iter = IterLoader(self.train_set_loader)
        self.val_set_iter = IterLoader(self.val_set_loader)

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

        # filtered_covid_others = FilterDataset(test_covid_dataset, labels=["Bacterial Pneumonia", "Fungal Pneumonia"])
        dataset_train_all = Merge_Dataset((filtered_covid, filtered_NIH), seed=1)
        # spilt dataset
        train_set, val_set = data.random_split(dataset_train_all, [int(dataset_train_all.length*5/6), int(dataset_train_all.length/6)])
        print('train:', len(train_set), 'validation:', len(val_set))
        return train_set, val_set

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

if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision.transforms as t

    # Configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default='/home/jingxiongli/PycharmProjects/lungDatasets/')

    config = parser.parse_args()
    transform = torchvision.transforms.Compose([XRayCenterCrop(),
                                        XRayResizer(224),
                                        ToPILImage(),
                                        t.Grayscale(num_output_channels=3)
                                        ])

    aug = torchvision.transforms.RandomApply([t.ColorJitter(brightness=0.5, contrast=0.7),
                                              t.RandomRotation(120),
                                              t.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.75, 1.33), interpolation=2),
                                              t.RandomHorizontalFlip(),
                                              t.RandomVerticalFlip(),
                                          ], p=0.5)
    lung_dataset = dataset_loader(config, transform, aug)


    for i, item in enumerate(lung_dataset.val_set):
        data = item['PA']
        label = item['lab']
        print('label:', label)
        plt.imshow(np.squeeze(data), cmap='gray')
        plt.show()
    print('0')