import csv
import os
try:
	import torch
except ImportError:
	print("no torch")
import logging
import sys

import utils


def create_folders_if_necessary(path):
	dirname = os.path.dirname(path)
	if not os.path.isdir(dirname):
		os.makedirs(dirname)


def get_storage_dir():
	if "RL_STORAGE" in os.environ:
		return os.environ["RL_STORAGE"]
	return "storage"


def get_model_dir(model_name):
	return os.path.join(get_storage_dir(), model_name)


def get_status_path(model_dir):
	return os.path.join(model_dir, "status.pt")


def get_status(model_dir):
	path = get_status_path(model_dir)
	return torch.load(path)


def save_status(status, model_dir):
	path = get_status_path(model_dir)
	utils.create_folders_if_necessary(path)
	torch.save(status, path)


def get_vocab(model_dir):
	return get_status(model_dir)["vocab"]


def get_model_state(model_dir):
	return get_status(model_dir)["model_state"]


def get_txt_logger(model_dir):
	path = os.path.join(model_dir, "log.txt")
	utils.create_folders_if_necessary(path)

	log = logging.getLogger('log')

	log.setLevel(logging.INFO)
	log.addHandler(logging.FileHandler(filename=path))
	log.addHandler(logging.StreamHandler(sys.stdout))
	
	return log


def get_csv_logger(model_dir):
	csv_path = os.path.join(model_dir, "log.csv")
	utils.create_folders_if_necessary(csv_path)
	csv_file = open(csv_path, "a")
	return csv_file, csv.writer(csv_file)

def get_csv_loss_logger(model_dir):
	csv_path = os.path.join(model_dir, "loss.csv")
	utils.create_folders_if_necessary(csv_path)
	csv_file = open(csv_path, "a")
	return csv_file, csv.writer(csv_file)

def get_txt_loss_logger(model_dir):
	path = os.path.join(model_dir, "loss.txt")
	utils.create_folders_if_necessary(path)

	loss_logger = logging.getLogger('loss')

	loss_logger.setLevel(logging.INFO)
	loss_logger.addHandler(logging.FileHandler(filename=path))

	return loss_logger