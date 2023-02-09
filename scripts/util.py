import os
import random
import numpy as np
import torch

""" Paths """
TRAIN_PATH = "new_data/train/"
TEST_PATH = "new_data/test/"
TRAIN_PATH_FRAMES = "new_data/train/frames/*"
TRAIN_PATH_MASKS = "new_data/train/masks/*"
VALID_PATH_FRAMES = "new_data/test/frames/*"
VALID_PATH_MASKS = "new_data/test/masks/*"


""" Define random seed """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


""" Create directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


""" Track time """
def epoch_time(start_time, end_time):
    total = end_time - start_time
    mins = int(total / 60)
    secs = int(total - (mins * 60))
    return mins, secs
