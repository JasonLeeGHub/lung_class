import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
from core.model import AttentionModule, res50Encoder, res50EncoderOnly
from core.utils import TopKAccuracyMetric, time_now, os_walk

class base:
    def __init__(self, config, loader):
        self.config = config

        # paths
        self.loader = loader

        self.save_path = config.save_path
        self.save_model_path = os.path.join(self.save_path, 'model/')
        self.save_features_path = os.path.join(self.save_path, 'features/')
        self.save_logs_path = os.path.join(self.save_path, 'logs/')
        self.save_results_path = os.path.join(self.save_path, 'results/')
        self.save_images_path = os.path.join(self.save_path, 'images/')

        # dataset configuration
        self.class_num = config.class_num


        # train configuration
        self.attention_map_num = config.attention_map_num
        self.base_learning_rate = config.base_learning_rate
        self.milestones = config.milestones
        self.device = torch.device('cuda')

        # test configuration

        # init_model
        ## the feature alignment module
        encoder = res50Encoder(config)
        # encoder = res50EncoderOnly(config)
        attention_module = AttentionModule(config)
        self.encoder = torch.nn.DataParallel(encoder).to(self.device)
        self.attention_module = torch.nn.DataParallel(attention_module).to(self.device)

        ## add all models to a list for esay using
        self.model_list = []
        self.model_list.append(self.encoder)
        self.model_list.append(self.attention_module)

        # init_criterions
        self.MSELoss = torch.nn.MSELoss()
        self.L1Loss = torch.nn.L1Loss()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

        # init_optimizer
        params = [{'params': self.encoder.parameters(), 'lr': self.base_learning_rate},
                  {'params': self.attention_module.parameters(), 'lr': self.base_learning_rate}]
        self.optimizer = optim.SGD(params=params, weight_decay=5e-4, momentum=0.9, nesterov=True)   
        self.ide_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, self.milestones, gamma=0.1)

    def lr_decay(self, current_step):
        self.ide_lr_scheduler.step(current_step)

    def compute_classification_loss(self, logits, pids):
        loss_i = self.CrossEntropyLoss(logits, pids)
        accuracy = TopKAccuracyMetric(topk=(1, 1))
        acc = accuracy(logits, pids)
        return acc, loss_i


    def save_model(self, save_epoch):

        # save model
        for ii, _ in enumerate(self.model_list):
            torch.save(self.model_list[ii].state_dict(),
                       os.path.join(self.save_model_path, 'model-{}_{}.pkl'.format(ii, save_epoch)))

        # if saved model is more than max num, delete the model with smallest epoch
        if self.config.max_save_model_num > 0:
            root, _, files = os_walk(self.save_model_path)

            # get indexes of saved models
            indexes = []
            for file in files:
                indexes.append(int(file.replace('.pkl', '').split('_')[-1]))

            # remove the bad-case and get available indexes
            model_num = len(self.model_list)
            available_indexes = copy.deepcopy(indexes)
            for element in indexes:
                if indexes.count(element) < model_num:
                    available_indexes.remove(element)

            available_indexes = sorted(list(set(available_indexes)), reverse=True)
            unavailable_indexes = list(set(indexes).difference(set(available_indexes)))

            # delete all unavailable models
            for unavailable_index in unavailable_indexes:
                try:
                    # os.system('find . -name "{}*_{}.pkl" | xargs rm  -rf'.format(self.config.save_models_path, unavailable_index))
                    for ii in range(len(self.model_list)):
                        os.remove(os.path.join(root, 'model-{}_{}.pkl'.format(ii, unavailable_index)))
                except:
                    pass

            # delete extra models
            if len(available_indexes) >= self.config.max_save_model_num:
                for extra_available_index in available_indexes[self.config.max_save_model_num:]:
                    # os.system('find . -name "{}*_{}.pkl" | xargs rm  -rf'.format(self.config.save_models_path, extra_available_index))
                    for ii in range(len(self.model_list)):
                        os.remove(os.path.join(root, 'model-{}_{}.pkl'.format(ii, extra_available_index)))


    ## resume model from resume_epoch
    def resume_model(self, resume_epoch):
        for ii, _ in enumerate(self.model_list):
            self.model_list[ii].load_state_dict(
                torch.load(os.path.join(self.save_model_path, 'model-{}_{}.pkl'.format(ii, resume_epoch))))
        print('Time: {}, successfully resume model from {}'.format(time_now(), resume_epoch))

    # def resume_model(self, resume_epoch):
    #     for ii, _ in enumerate(self.model_list):
    #         model_dict = self.model_list[ii].state_dict()  # 自己的模型参数变量
    #         pretrained_dict = torch.load(os.path.join(self.save_model_path, 'model-{}_{}.pkl'.format(ii, resume_epoch)))
    #         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 去除一些不需要的参数
    #         model_dict.update(pretrained_dict)  # 参数更新
    #         # model.load_state_dict(model_dict)  # 加载
    #
    #         self.model_list[ii].load_state_dict(model_dict)
    #     print('Time: {}, successfully resume model from {}'.format(time_now(), resume_epoch))


    ## resume model from resume_epoch
    def resume_model_from_path(self, path, resume_epoch):
        for ii, _ in enumerate(self.model_list):
            self.model_list[ii].load_state_dict(
                torch.load(os.path.join(path, 'model-{}_{}.pkl'.format(ii, resume_epoch))))
        print('Time: {}, successfully resume model from {}'.format(time_now(), resume_epoch))


    ## set model as train mode
    def set_train(self):
        for ii, _ in enumerate(self.model_list):
            self.model_list[ii] = self.model_list[ii].train()


    ## set model as eval mode
    def set_eval(self):
        for ii, _ in enumerate(self.model_list):
            self.model_list[ii] = self.model_list[ii].eval()