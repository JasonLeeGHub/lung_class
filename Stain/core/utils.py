import numpy as np
import time
import torch
import torch.nn.functional as F
import random
import os


class Metric(object):
    pass

class TopKAccuracyMetric(Metric):
    def __init__(self, topk=(1,)):
        self.name = 'topk_accuracy'
        self.topk = topk
        self.maxk = max(topk)
        self.reset()

    def reset(self):
        self.corrects = np.zeros(len(self.topk))
        self.num_samples = 0.

    def __call__(self, output, target):
        """Computes the precision@k for the specified values of k"""
        self.num_samples += target.size(0)
        _, pred = output.topk(self.maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for i, k in enumerate(self.topk):
            correct_k = correct[:k].view(-1).float().sum(0)
            self.corrects[i] += correct_k.item()

        return self.corrects * 100. / self.num_samples

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

        result += str(loss_name)
        result += ': '
        result += str(loss_value)
        result += ';  '

    return result

def analyze_meter_4_csv(loss_names, loss_meter):

    result = []
    for i in range(len(loss_names)):

        loss_value = round(loss_meter[i],3)
        result.append(loss_value)

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

class CatMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None

    def update(self, val):
        # print(val.shape)
        if self.val is None:
            self.val = val
        else:
            self.val = np.concatenate([self.val, val], axis=0)
    def get_val(self):
        return self.val

    def get_val_numpy(self):
        return self.val.data.cpu().numpy()

def batch_augment(images, attention_map, mode='mixup', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()

    if mode == 'mixup':
        auged_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            mixup_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c
            nonzero_indices = torch.nonzero(mixup_mask[0, 0, ...])
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)
            upsampled_patch = F.upsample_bilinear(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW))
            auged_image = images[batch_index:batch_index + 1,:,:,:]*0.6 + upsampled_patch*0.4
            # import matplotlib.pyplot as plt
            # plt.subplot(2,1,1)
            # plt.imshow(upsampled_patch[0][0].squeeze().cpu().numpy(), cmap='gray')
            # plt.subplot(2,1,2)
            # plt.imshow(auged_image[0][0].squeeze().cpu().numpy(), cmap='gray')
            # plt.show()


            auged_images.append(auged_image)
        auged_images = torch.cat(auged_images, dim=0)
        return auged_images

    elif mode == 'crop':
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

    elif mode == 'patch':
        multi_image = []
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
            patch = images.clone()[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max]
            auged_image = images.clone()[batch_index:batch_index + 1, :, ...]
            H_patch = random.randint(0, imgH-(height_max-height_min))
            W_patch = random.randint(0, imgW-(width_max-width_min))
            auged_image[:, :,H_patch:H_patch+(height_max-height_min), W_patch:W_patch+(width_max-width_min)] = patch
            multi_image.append(auged_image)
            # import matplotlib.pyplot as plt
            # plt.subplot(2,1,1)
            # plt.imshow(patch[0][0].squeeze().cpu().numpy(), cmap='gray')
            # plt.subplot(2,1,2)
            # plt.imshow(auged_image[0][0].squeeze().cpu().numpy(), cmap='gray')
            # plt.show()
        multi_images = torch.cat(multi_image, dim=0)
        return multi_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)

