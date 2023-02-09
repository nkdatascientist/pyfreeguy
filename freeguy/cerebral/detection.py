
from freeguy.utils import check_class_accuracy, mean_average_precision
from freeguy.utils import AverageMeter, get_evaluation_bboxes
from tqdm import tqdm

import torch
import warnings
warnings.filterwarnings("ignore")

def train_model():
    
    train_loss = AverageMeter("Train_Loss", ":6.6f")
    running_loss = AverageMeter("Running_Loss", ":6.6f")


def val_model():
    return 