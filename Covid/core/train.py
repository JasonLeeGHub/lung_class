# import sys
# sys.path.append('../')
import os
import torch
import torch.nn.functional as F
from .utils import AverageMeter, time_now
from .base import base


def train_a_ep(config, base, loader, current_step):

    # set train mode and learning rate decay
    base.set_train()
    meter = AverageMeter()
    val_meter = AverageMeter()
    for iteration in range(config.ep_size):
        titles, values = train_a_iter(config, base, loader, current_step)
        meter.update(values, 1)
    base.lr_decay(current_step)
    base.set_eval()
    for iteration in range(10):
        val_titles, val_values = validate_a_iter(config, base, loader, current_step)
        val_meter.update(val_values, 1)

    return titles, meter.get_val_numpy(), val_titles, val_meter.get_val_numpy()

def train_a_iter(config, base, loader, current_step):
    ### load data
    data_next = loader.train_set_iter.next_one()
    img_input, pid = data_next['PA'], data_next['lab']
    img_input, pid = img_input.to(base.device), pid.long().to(base.device)
    ### forward1
    feature_4_attention, logit_raw = base.encoder(img_input)
    ### attention
    attention_map, attention_map_sum = base.attention_module(feature_4_attention)
    ### reroll
    img_reroll = img_input * attention_map_sum
    _, logit_attention = base.encoder(img_reroll)

    ### loss
    acc_raw, loss_raw = base.compute_classification_loss(logit_raw, pid)
    # print(acc_raw, loss_raw)
    acc_attention, loss_attention = base.compute_classification_loss(logit_attention, pid)
    loss = loss_raw + loss_attention
    print(time_now(), acc_raw[0], loss_raw.data, acc_attention[0], loss_attention.data)
    base.optimizer.zero_grad()
    loss.backward()
    base.optimizer.step()

    return ['acc_raw', 'loss_raw', 'acc_attention','loss_attention'], \
	       torch.Tensor([acc_raw[0], loss_raw.data, acc_attention[0], loss_attention.data])


def validate_a_iter(config, base, loader, current_step):
    with torch.no_grad():

        ### load data
        data_next = loader.val_set_iter.next_one()
        img_input, pid = data_next['PA'], data_next['lab']
        img_input, pid = img_input.to(base.device), pid.long().to(base.device)
        ### forward1
        feature_4_attention, logit_raw = base.encoder(img_input)
        ### attention
        attention_map, attention_map_sum = base.attention_module(feature_4_attention)
        ### reroll
        img_reroll = img_input * attention_map_sum
        _, logit_attention = base.encoder(img_reroll)
        ### loss
        acc_raw, loss_raw = base.compute_classification_loss(logit_raw, pid)
        acc_attention, loss_attention = base.compute_classification_loss(logit_attention, pid)

    print(time_now(), acc_raw[0], loss_raw.data, acc_attention[0], loss_attention.data)

    return ['val_acc_raw', 'val_loss_raw', 'val_acc_attention', 'val_loss_attention'], \
           torch.Tensor([acc_raw[0], loss_raw.data, acc_attention[0], loss_attention.data])