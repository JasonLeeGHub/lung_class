import torchvision
import torchvision.transforms as t
import os
import copy
import argparse
import numpy as np
from dataset.Dataloader import dataset_loader
from dataset.covid19 import XRayCenterCrop, XRayResizer, ToPILImage, histeq
from core.base import base
from core.utils import make_dirs, Logger, os_walk, time_now, analyze_names_and_meter
from core.train import train_a_ep

def main(config):

	# environments
	make_dirs(config.save_path)
	make_dirs(os.path.join(config.save_path, 'logs/'))
	make_dirs(os.path.join(config.save_path, 'model/'))
	make_dirs(os.path.join(config.save_path, 'features/'))
	make_dirs(os.path.join(config.save_path, 'results/'))
	make_dirs(os.path.join(config.save_path, 'images/'))
	os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'


	# loaders
    # data_aug = transforms.Compose([
    #
    # ])
	# transform_train = transforms.Compose(
    #     [XRayCenterCrop(),
    #      XRayResizer(config.image_size),
	# 	 ToPILImage(),
	# 	 histeq(),
	# 	 transforms.Grayscale(num_output_channels=3),
	# 	 transforms.ToTensor()])

	transform = torchvision.transforms.Compose([XRayCenterCrop(),
												XRayResizer(224),
												ToPILImage(),
												t.Grayscale(num_output_channels=3)
												])

	aug = torchvision.transforms.RandomApply([t.ColorJitter(brightness=0.5, contrast=0.7),
											  t.RandomRotation(120),
											  t.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.75, 1.33),
																  interpolation=2),
											  t.RandomHorizontalFlip(),
											  t.RandomVerticalFlip(),
											  ], p=0.5)
	aug = t.Compose([aug, t.ToTensor()])

	loader= dataset_loader(config, transform, aug)

	# base
	Base = base(config, loader)


	# logger
	logger = Logger(os.path.join(os.path.join(config.save_path, 'logs/'), 'logging.txt'))
	logger(config)


	if config.mode == 'train':

		# automatically resume model from the latest one
		start_train_epoch = 0
		if True:
			root, _, files = os_walk(Base.save_model_path)
			if len(files) > 0:
				# get indexes of saved models
				indexes = []
				for file in files:
					indexes.append(int(file.replace('.pkl', '').split('_')[-1]))

				# remove the bad-case and get available indexes
				model_num = len(Base.model_list)
				available_indexes = copy.deepcopy(indexes)
				for element in indexes:
					if indexes.count(element) < model_num:
						available_indexes.remove(element)

				available_indexes = sorted(list(set(available_indexes)), reverse=True)
				unavailable_indexes = list(set(indexes).difference(set(available_indexes)))

				if len(available_indexes) > 0:  # resume model from the latest model
					Base.resume_model(available_indexes[0])
					start_train_epoch = available_indexes[0]
					logger('Time: {}, automatically resume training from the latest step (model {})'.
						   format(time_now(), available_indexes[0]))
				else:  #
					logger('Time: {}, there are no available models')

		# train loop
		for current_step in range(start_train_epoch, config.joint_training_steps):

			# save model every step. extra models will be automatically deleted for saving storage
			Base.save_model(current_step)

			# evaluate reid
			# if (current_step+1)%10 ==0:
			# 	logger('**********' * 10 + 'evaluate' + '**********' * 10)
			# 	results = test(config, base, loader_target, True)
			# 	for key in list(results.keys()):
			# 		logger('Time: {}, {}, {}'.format(time_now(), key, results[key]))
			# 	logger('')
			logger('**********'*10 + 'train' + '**********'*10 )
			train_titles, train_values, val_titles, val_values = train_a_ep(config, Base, loader, current_step)
			logger('Time: {};  Step: {};  {}'.format(time_now(), current_step, analyze_names_and_meter(train_titles, train_values)))
			logger('Time: {};  Step: {};  {}'.format(time_now(), current_step, analyze_names_and_meter(val_titles, val_values)))
			logger('')



	# elif config.mode == 'test':
    #
	# 	base.resume_model_from_path(config.pretrained_model_path, config.pretrained_model_index)
	# 	logger('**********' * 10 + 'test with pre-trained model' + '**********' * 10)
	# 	results = test(config, base, loaders, False)
	# 	for key in list(results.keys()):
	# 		logger('Time: {}, {}, {}'.format(time_now(), key, results[key]))
	# 	logger('')

if __name__ == '__main__':


	# Configurations
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type=str, default='train')

	# output configuration
	parser.add_argument('--save_path', type=str, default='out/base/', help='path to save models, logs, images')
	# dataset configuration
	parser.add_argument('--dataset_path', type=str,
						default='/home/jingxiongli/PycharmProjects/lungDatasets/')
	parser.add_argument('--class_num', type=int, default=2, help='identity numbers in training set')
	parser.add_argument('--attention_map_num', type=int, default=32, help='attention map numbers in training set')
	parser.add_argument('--image_size', type=int, default=224, help='image size for pixel alignment module,. in feature alignment module, images will be automatically reshaped to 384*192')



	# training configuration
	parser.add_argument('--base_learning_rate', type=float, default=0.002, help='learning rate for feature alignment module')
	parser.add_argument('--ep_size', type=int, default=100, help='how many iters for a ep')

	# training configuration
	parser.add_argument('--joint_training_steps', type=int, default=400)
	parser.add_argument('--milestones', nargs='+', type=int, default=[50])
	parser.add_argument('--save_model_steps', nargs='+', type=int, default=[100])

	# evaluate configuration
	parser.add_argument('--max_save_model_num', type=int, default=2, help='0 for max num is infinit, extra models will be automatically deleted for saving storage')

	# test configuration
	parser.add_argument('--pretrained_model_path', type=str, default='', help='please download the pretrained model at first, and then set path')
	parser.add_argument('--pretrained_model_index', type=int, default=None, help='')


	# parse
	config = parser.parse_args()
	config.milestones = list(np.array(config.milestones))
	config.save_model_steps = list(np.array(config.save_model_steps))

	main(config)