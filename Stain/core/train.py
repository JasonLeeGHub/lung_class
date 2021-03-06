# import sys
# sys.path.append('../')
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .utils import AverageMeter, time_now, batch_augment, CatMeter
from sklearn.metrics import precision_recall_curve, average_precision_score,roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix
from .base import base


def test_a_ep(config, base, loader, current_step):
    test_meter = AverageMeter()
    test_score_meter = CatMeter()
    test_pid_meter = CatMeter()
    test_pred_meter = CatMeter()

    base.set_eval()
    for iteration in range(30):
        test_titles, test_values, test_score, test_pid, test_pred = test_a_iter(config, base,
                                                                              loader.test_iter,
                                                                              current_step)
        test_pred_meter.update(test_pred)
        test_meter.update(test_values, 1)
        test_score_meter.update(test_score)
        test_pid_meter.update(test_pid)


    score = test_score_meter.get_val()
    pred = test_pred_meter.get_val()
    pid = test_pid_meter.get_val()

    confusion = confusion_matrix(pid, pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    Sensitivity = TP / float(TP + FN)
    Specificity = TN / float(TN + FP)
    Precision = precision_score(pid, pred)
    Recall = recall_score(pid, pred)
    F1 = f1_score(pid, pred)

    precision, recall, _ = precision_recall_curve(pid, score)
    ap = average_precision_score(pid, score)
    fpr, tpr, _ = roc_curve(pid, score)
    plt.figure(0)
    plt.plot(recall, precision, 'k--', color=(0.1, 0.9, 0.1), label='AP = {0:.2f}'.format(ap), lw=2)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Recall Rate")
    plt.ylabel("Precision Rate")
    plt.legend(loc="lower right")
    fig_name = 'PR_test' + str(current_step) + '.jpg'
    plt.savefig(os.path.join(config.save_path, 'images', fig_name))
    plt.clf()

    plt.figure(1)
    Auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'k--', color=(0.1, 0.1, 0.9), label='Mean ROC (area = {0:.2f})'.format(Auc), lw=2)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  # 画对角线
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    fig_name = 'ROC_test' + str(current_step) + '.jpg'
    plt.savefig(os.path.join(config.save_path, 'images', fig_name))
    plt.clf()
    metric_bag = [ap, Auc, Precision, Recall, Sensitivity, Specificity, F1]
    return test_titles, test_meter.get_val_numpy(),metric_bag





def train_a_ep(config, base, loader, current_step):
    # set train mode and learning rate decay
    train_meter = AverageMeter()
    val_meter = AverageMeter()
    score_meter = CatMeter()
    pid_meter = CatMeter()
    pred_meter = CatMeter()


    for i in range(5):
        train_titles, train_meter, val_titles, val_meter, score_meter, pid_meter, pred_meter = train_a_fold(config, base, loader,
                                                                                                current_step,
                                                                                                train_meter, val_meter,
                                                                                                score_meter, pid_meter,
                                                                                                pred_meter,
                                                                                                i+1)
        print('finish fold ', i)

    score =score_meter.get_val()
    pred =pred_meter.get_val()
    pid = pid_meter.get_val()

    confusion = confusion_matrix(pid, pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    Sensitivity = TP / float(TP + FN)
    Specificity = TN / float(TN + FP)
    Precision = precision_score(pid, pred)
    Recall = recall_score(pid, pred)
    F1 = f1_score(pid, pred)


    precision, recall, _ = precision_recall_curve(pid, score)
    ap = average_precision_score(pid, score)
    fpr, tpr, _ = roc_curve(pid, score)
    plt.figure(0)
    plt.plot(recall, precision,'k--', color=(0.1,0.9,0.1),label='AP = {0:.2f}'.format(ap), lw=2)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Recall Rate")
    plt.ylabel("Precision Rate")
    plt.legend(loc="lower right")
    fig_name = 'PR_' + str(current_step) + '.jpg'
    plt.savefig(os.path.join(config.save_path,'images', fig_name))
    plt.clf()


    plt.figure(1)
    Auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'k--', color=(0.1,0.1,0.9), label='Mean ROC (area = {0:.2f})'.format(Auc), lw=2)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  # 画对角线
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    fig_name = 'ROC_' + str(current_step) + '.jpg'
    plt.savefig(os.path.join(config.save_path,'images', fig_name))
    plt.clf()
    metric_bag = [ap,Auc,Precision,Recall,Sensitivity,Specificity,F1]
    # return train_titles, train_meter.get_val_numpy(), _, _,test_titles, test_meter.get_val_numpy(),metric_bag
    return train_titles, train_meter.get_val_numpy(), val_titles, val_meter.get_val_numpy(),_, _,metric_bag

def train_a_fold(config, base, loader, current_step, train_meter, val_meter,score_meter, pid_meter, pred_meter,fold_name):
    base.set_train()
    for iteration in range(config.ep_size):
        train_titles, train_values = train_a_iter(config, base, eval(('loader.train_iter_'+str(fold_name))), current_step)
        train_meter.update(train_values, 1)
    base.lr_decay(current_step)
    base.set_eval()
    for iteration in range(5):
        val_titles, val_values, score, pid, pred = validate_a_iter(config, base, eval('loader.val_iter_'+str(fold_name)), current_step)

        # val_titles, val_values, score, pid, pred = test_a_iter_res50_only(config, base,
        #                                                                     loader.test_iter,
        #                                                                    current_step)
        pred_meter.update(pred)
        val_meter.update(val_values, 1)
        score_meter.update(score)
        pid_meter.update(pid)
    return train_titles, train_meter, val_titles, val_meter, score_meter,pid_meter, pred_meter


def train_a_iter(config, base, loader, current_step):
    base.optimizer.zero_grad()
    import matplotlib.pyplot as plt
    ### load data
    data_next = loader.next_one()
    img_input, pid = data_next['PA'], data_next['lab']
    img_input, pid = img_input.to(base.device), pid.long().to(base.device)
    ### forward1
    logit_raw, feature_1,attention_map = base.encoder(img_input)
    ### batch augs
    ## mixup and forward2
    mixup_images = batch_augment(img_input, attention_map[:, 0:1, :, :], mode='mixup', theta=(0.4, 0.6), padding_ratio=0.1)
    logit_mixup, _, _ = base.encoder(mixup_images)

    ## dropping
    drop_images = batch_augment(img_input, attention_map[:, 1:2, :, :], mode='drop', theta=(0.2, 0.5))
    logit_drop, _, _ = base.encoder(drop_images)

    ## patching
    patch_images = batch_augment(img_input, attention_map[:, 2:3, :, :], mode='patch', theta=(0.4, 0.6), padding_ratio=0.1)
    logit_patch, _, _= base.encoder(patch_images)

    # import matplotlib.pyplot as plt
    # for i in range(10):
    #     print(pid[i])
    #     print(logit_raw[i])
    #     CXR_img = img_input[i,0,:,:].detach().cpu().numpy()
    #     attention_map_1 = F.interpolate(attention_map[i:i+1,0:1,:,:], size=224, mode='bilinear').squeeze().detach().cpu().numpy()
    #     attention_map_2 = F.interpolate(attention_map[i:i+1,1:2,:,:], size=224, mode='bilinear').squeeze().detach().cpu().numpy()
    #     attention_map_3 = F.interpolate(attention_map[i:i+1,2:,:,:], size=224, mode='bilinear').squeeze().detach().cpu().numpy()
    #     mixup_img = mixup_images[i,0,:,:].detach().cpu().numpy()
    #     drop_img = drop_images[i, 0, :, :].detach().cpu().numpy()
    #     patch_img = patch_images[i, 0, :, :].detach().cpu().numpy()
    #     plt.subplot(2,4,1)
    #     plt.imshow(CXR_img, cmap='gray')
    #     plt.axis('off')
    #     plt.subplot(2,4,2)
    #     plt.imshow(attention_map_1)
    #     plt.axis('off')
    #     plt.subplot(2, 4, 3)
    #     plt.imshow(attention_map_2)
    #     plt.axis('off')
    #     plt.subplot(2, 4, 4)
    #     plt.imshow(attention_map_3)
    #     plt.axis('off')
    #     plt.subplot(2, 4, 5)
    #     plt.imshow(mixup_img, cmap='gray')
    #     plt.axis('off')
    #     plt.subplot(2, 4, 6)
    #     plt.imshow(drop_img, cmap='gray')
    #     plt.axis('off')
    #     plt.subplot(2, 4, 7)
    #     plt.imshow(patch_img, cmap='gray')
    #     plt.axis('off')
    #
    #     plt.show()



    ### loss
    acc_raw, loss_raw = base.compute_classification_loss(logit_raw, pid)
    acc_mixup, loss_mixup = base.compute_classification_loss(logit_mixup, pid)
    acc_drop, loss_drop = base.compute_classification_loss(logit_drop, pid)
    acc_patch, loss_patch = base.compute_classification_loss(logit_patch, pid)
    loss = (loss_raw + loss_mixup + loss_drop + loss_patch)/4
    # loss = loss_raw
    loss.backward()
    base.optimizer.step()
    return ['acc_raw', 'loss_raw', 'acc_mixup','loss_mixup','acc_drop','loss_drop','acc_patch','loss_patch',], \
	       torch.Tensor([acc_raw[0], loss_raw.data, acc_mixup[0], loss_mixup.data, acc_drop[0], loss_drop.data, acc_patch[0], loss_patch.data])
    # return ['acc_raw', 'loss_raw',], \
    #        torch.Tensor([acc_raw[0], loss_raw.data])



def validate_a_iter(config, base, loader, current_step):
    with torch.no_grad():
        ### load data
        data_next = loader.next_one()
        img_input, pid = data_next['PA'], data_next['lab']
        img_input, pid = img_input.to(base.device), pid.long().to(base.device)
        ### forward1
        logit_raw, feature_1,attention_map = base.encoder(img_input)
        mixup_images = batch_augment(img_input, attention_map[:, :1, :, :], mode='mixup', theta=(0.4, 0.6),
                                     padding_ratio=0.1)
        logit_mixup, _, _ = base.encoder(mixup_images, False)
        ### loss
        acc_raw, loss_raw = base.compute_classification_loss(logit_raw, pid)
        acc_attention, loss_attention = base.compute_classification_loss(logit_mixup,  pid)
        acc = (acc_raw[0] + acc_attention[0])/2
        ### metrics
        logit_mean = (logit_raw + logit_mixup)/2
        score_4sk = []
        _,pred = logit_mean.topk(1, 1, True, True)
        pred_4sk = pred.t().detach().cpu().numpy()[0]
        pid_4sk = pid.detach().cpu().numpy()
        for i, score_id in  enumerate(pid_4sk):
                score_4sk.append(logit_mean[i][1].detach().cpu().numpy())
        score_4sk = np.array(score_4sk)
    return ['val_acc_raw', 'val_loss_raw', 'val_acc_attention', 'val_loss_attention','acc'], \
           torch.Tensor([acc_raw[0], loss_raw.data, acc_attention[0], loss_attention.data, acc]),\
            score_4sk,pid_4sk, pred_4sk

def test_a_iter(config, base, loader, current_step):
    with torch.no_grad():
        ### load data
        data_next = loader.next_one()
        img_input, pid = data_next['PA'], data_next['lab']
        # print('pid:', pid)
        img_input, pid = img_input.to(base.device), pid.long().to(base.device)
        ### forward1
        logit_raw, feature_1, attention_map = base.encoder(img_input, False)
        # print('attensize:', attention_map.size())




        mixup_images = batch_augment(img_input, attention_map[:, :1, :, :], mode='mixup', theta=(0.4, 0.6),
                                     padding_ratio=0.1)
        logit_mixup, _, _ = base.encoder(mixup_images, False)
        ### loss
        acc_raw, loss_raw = base.compute_classification_loss(logit_raw, pid)
        # print(acc_raw)
        acc_attention, loss_attention = base.compute_classification_loss(logit_mixup,  pid)
        acc = (acc_raw[0] + acc_attention[0])/2
        ### metrics
        logit_mean = (logit_raw + logit_mixup)/2
        score_4sk = []
        _,pred = logit_mean.topk(1, 1, True, True)

        # import matplotlib.pyplot as plt
        # for i in range(10):
        #     print(pid[i])
        #     print(pred[i])
        #     CXR_img = img_input[i,0,:,:].detach().cpu().numpy()
        #     attention_map_1 = F.interpolate(attention_map[i:i+1,0:1,:,:], size=224, mode='bilinear').squeeze().detach().cpu().numpy()
        #     plt.subplot(2,2,1)
        #     plt.imshow(CXR_img, cmap='gray')
        #     plt.subplot(2,2,2)
        #     plt.imshow(attention_map_1,)
        #
        #     plt.show()


        pred_4sk = pred.t().detach().cpu().numpy()[0]
        # print(pred_4sk)
        pid_4sk = pid.detach().cpu().numpy()
        # print(pid_4sk)
        for i, score_id in  enumerate(pid_4sk):
                score_4sk.append(logit_mean[i][1].detach().cpu().numpy())
        score_4sk = np.array(score_4sk)
    return ['test_acc_raw', 'test_loss_raw', 'test_acc_attention', 'test_loss_attention','acc'], \
           torch.Tensor([acc_raw[0], loss_raw.data, acc_attention[0], loss_attention.data, acc]),\
            score_4sk,pid_4sk, pred_4sk



def train_a_iter_res50only(config, base, loader, current_step):
    ### load data
    data_next = loader.next_one()
    img_input, pid = data_next['PA'], data_next['lab']
    img_input, pid = img_input.to(base.device), pid.long().to(base.device)
    ### forward1
    logit_raw = base.encoder(img_input)

    ### loss
    acc_raw, loss_raw = base.compute_classification_loss(logit_raw, pid)

    loss = (loss_raw )
    base.optimizer.zero_grad()
    loss.backward()
    base.optimizer.step()

    return ['acc_raw', 'loss_raw',], \
	       torch.Tensor([acc_raw[0], loss_raw.data])


def validate_a_iter_res50_only(config, base, loader, current_step):
    with torch.no_grad():
        ### load data
        data_next = loader.next_one()
        img_input, pid = data_next['PA'], data_next['lab']
        img_input, pid = img_input.to(base.device), pid.long().to(base.device)
        ### forward1
        logit_raw = base.encoder(img_input)
        ### loss
        acc_raw, loss_raw = base.compute_classification_loss(logit_raw, pid)
        ### metrics
        logit_mean = logit_raw
        score_4sk = []
        _,pred = logit_mean.topk(1, 1, True, True)
        pred_4sk = pred.t().detach().cpu().numpy()[0]
        pid_4sk = pid.detach().cpu().numpy()
        for i, score_id in  enumerate(pid_4sk):
                score_4sk.append(logit_mean[i][1].detach().cpu().numpy())
        score_4sk = np.array(score_4sk)

    # print(time_now(), acc_raw[0], loss_raw.data, acc_attention[0], loss_attention.data)

    return ['val_acc_raw', 'val_loss_raw'], \
           torch.Tensor([acc_raw[0], loss_raw.data,]),\
            score_4sk,pid_4sk, pred_4sk


def test_a_iter_res50_only(config, base, loader, current_step):
    with torch.no_grad():
        ### load data
        data_next = loader.next_one()
        img_input, pid = data_next['PA'], data_next['lab']
        img_input, pid = img_input.to(base.device), pid.long().to(base.device)
        ### forward1
        logit_raw = base.encoder(img_input)
        ### loss
        acc_raw, loss_raw = base.compute_classification_loss(logit_raw, pid)
        ### metrics
        logit_mean = logit_raw
        score_4sk = []
        _,pred = logit_mean.topk(1, 1, True, True)
        pred_4sk = pred.t().detach().cpu().numpy()[0]
        pid_4sk = pid.detach().cpu().numpy()
        for i, score_id in  enumerate(pid_4sk):
                score_4sk.append(logit_mean[i][1].detach().cpu().numpy())
        score_4sk = np.array(score_4sk)

    # print(time_now(), acc_raw[0], loss_raw.data, acc_attention[0], loss_attention.data)

    return ['test_acc_raw', 'test_loss_raw'], \
           torch.Tensor([acc_raw[0], loss_raw.data,]),\
           score_4sk,pid_4sk, pred_4sk