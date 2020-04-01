import numpy as np
import time
import torch
import torch.nn.functional as F
import random
import os

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def time_now():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files

def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Successfully make dirs: {}'.format(dir))
    else:
        print('Existed dirs: {}'.format(dir))

def analyze_names_and_meter(loss_names, loss_meter):

    result = ''
    for i in range(len(loss_names)):

        loss_name = loss_names[i]
        loss_value = loss_meter[i]

        result += loss_name
        result += ': '
        result += str(loss_value)
        result += ';  '

    return result
## logger
class Logger:

    def __init__(self, logger_path):
        self.logger_path = logger_path

    def __call__(self, input, newline=True):
        input = str(input)
        if newline:
            input += '\n'

        with open(self.logger_path, 'a') as f:
            f.write(input)
            f.close()

        print(input)

# Meters
class AverageMeter:

    def __init__(self, neglect_value=None):
        self.reset()
        self.neglect_value = neglect_value

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n):
        if self.neglect_value is None or self.neglect_value not in val:
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    def get_val(self):
        return self.avg

    def get_val_numpy(self):
        return self.avg.data.cpu().numpy()



def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            crop_images.append(
                F.upsample_bilinear(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)

