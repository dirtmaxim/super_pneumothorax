import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable
import segmentation_models_pytorch as smp
from utils import run_length_decode, run_length_encode
from losses import *
from metrics import DiceCompCoef
from tqdm import tqdm as tqdm
from tensorboardX import SummaryWriter
import time
import sys
import pandas as pd
from augs import weak_aug, val_aug
from my_datasets import *
import yaml
import skimage
from torch.optim import *
from torch.optim.lr_scheduler import *
from scipy import ndimage
import torch.nn as nn
import collections
from apex import amp

use_amp = True

def horiz_flip(img):
    return cv2.flip(img, 1)

def rot_pos15(img):
    return ndimage.rotate(img, 15, reshape=False, mode='nearest')

def rot_neg15(img):
    return ndimage.rotate(img, -15, reshape=False, mode='nearest')

def filter_mask_instances(mask, instance_threshold):
    passed_labels = []
    labeled_mask = skimage.measure.label(mask)
    unique, counts = np.unique(labeled_mask, return_counts=True)
    for label in unique[1:]:
        if counts[label] > instance_threshold:
            passed_labels.append(label)
    for p_label in passed_labels:
        mask[mask == p_label] == 1
    if len(passed_labels) == 0:
        mask = np.zeros(mask.shape)
    return mask

def remove_small_rle_enc(pred, size_thr, sigm_thr):
    pred = filter_mask_instances(pred >= sigm_thr, size_thr)
    if pred.sum() > 0:
        subm_str = run_length_encode(pred >= sigm_thr)
    else:
        subm_str = -1
    return subm_str

def train_model(model, dataloader, optimizer, criterion, scheduler, epoch, batch_sch_step = False, device = None):
    epoch_len = len(dataloader)
    model.train()
    batch_losses = []
    lrs = []
    with tqdm(total=epoch_len, file=sys.stdout) as pbar:
        for idx, batch_sampled in enumerate(dataloader):
            if batch_sch_step:
                scheduler.step()
            for param_group in optimizer.param_groups:
                lrs.append(param_group['lr'])
            inputs_cpu = batch_sampled[0]
            targets_cpu = batch_sampled[1]
            inputs = inputs_cpu.to(device, non_blocking=True)
            targets = targets_cpu.to(device, non_blocking=True)

            #inputs = inputs_cpu.cuda(non_blocking=True)
            #targets = targets_cpu.cuda(non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model.forward(inputs)

            loss = criterion(outputs, targets)

            if use_amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            pbar.set_description('L: %f' % batch_loss)
            pbar.update(1)

    return batch_losses,lrs

def validate_model(model, dataloader, criterion, dice_coef, dice_comp_coef, device = None):
    val_loss, dice, dice_comp, pos_cnt = 0,0,0,0
    TP, FP, TN, FN = 0,0,0,0
    epoch_len = len(dataloader)
    model.eval()
    with torch.no_grad() as tn:
        for idx, batch_sampled in tqdm(enumerate(dataloader)):
            inputs = Variable(batch_sampled[0].to(device))
            targets = Variable(batch_sampled[1].to(device))
            outputs = model.forward(inputs)
            val_loss += criterion(outputs, targets).item()
            tar_sum = targets.sum()
            out_sum = (torch.sigmoid(outputs) >= 0.5).sum()
            
            if (tar_sum == 0) and (out_sum == 0):
                TN+=1
            elif (tar_sum == 0) and (out_sum != 0):
                FP+=1
            
            if tar_sum > 0 :
                dice += dice_coef(outputs, targets).item()
                pos_cnt+=1
            dice_comp += dice_comp_coef(outputs, targets)
    return (val_loss / epoch_len), (dice / pos_cnt), (dice_comp / epoch_len).cpu().numpy(), (FP / (FP + TN))

def test_model(model, dataloader, def_size=1024, device=None):
    epoch_len = len(dataloader)
    model.eval()
    preds = np.zeros((len(dataloader),def_size,def_size), dtype=np.float32)
    with torch.no_grad() as tn:
        for idx, batch_sampled in tqdm(enumerate(dataloader)):
            inputs = Variable(batch_sampled[0].to(device))
            outputs = model.predict(inputs)
            pred = outputs[0,0].cpu().numpy()
            pred = cv2.resize(pred, (def_size,def_size))
            preds[idx] = pred
    return preds

def tta_predictions(model, dataloader, def_size=1024, device=None):
    tta_in = [None,horiz_flip]
    tta_out = [None,horiz_flip]
    cumul_preds = np.zeros((len(dataloader),def_size,def_size), dtype=np.float32)
    for idx,tta_func in enumerate(tta_in):
        dataloader.dataset.tta_func = tta_func
        cur_preds = test_model(model, dataloader, device=device)
        if tta_func is not None:
            for i,xi in enumerate(cur_preds):
                cur_preds[i] = tta_out[idx](xi)
        cumul_preds += cur_preds
        cur_preds = None
        cur_preds_tr = None
    cumul_preds /= len(tta_in)
    return cumul_preds

def postproc_n_convert(cumul_preds, filenames, instance_thr = 2400, sigm_thr = 0.5):
    rle_preds = [remove_small_rle_enc(xi, instance_thr, sigm_thr) for xi in cumul_preds]
    df = pd.DataFrame(filenames, columns=['ImageId'])
    df['EncodedPixels'] = rle_preds
    return df

class SegTrainer:
    def __init__(self, conf_path, args):
        with open(conf_path, 'r') as stream:
            try:
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.IMG_SIZE = config_dict['IMG_SIZE']
        self.DEF_SIZE = config_dict['DEF_SIZE']
        self.MEAN=tuple(config_dict['MEAN'])
        self.STD=tuple(config_dict['STD'])
        self.BATCH_SIZE=config_dict['BATCH_SIZE']
        self.N_WORKERS=config_dict['N_WORKERS']
        self.N_EPOCHS = config_dict['N_EPOCHS']
        self.LOGS_PATH = config_dict['LOGS_PATH']
        self.SNAPSHOTS_PATH = config_dict['SNAPSHOTS_PATH']
        self.MODEL_ALIAS = config_dict['MODEL_ALIAS']
        self.TRAIN_IMG_ROOT = config_dict['TRAIN_IMG_ROOT']
        self.TEST_IMG_ROOT = config_dict['TEST_IMG_ROOT']
        self.ANN_PICKLE_PATH = config_dict['ANN_PICKLE_PATH']
        self.TRAIN_PICKLE_PATH = config_dict['TRAIN_PICKLE_PATH']
        self.VAL_PICKLE_PATH = config_dict['VAL_PICKLE_PATH']
        self.TEST_PICKLE_PATH = config_dict['TEST_PICKLE_PATH']
        self.DEVICE = config_dict['DEVICE']
        
        dataset_class = globals()[config_dict['dataset_class']]
        self.train_dataset = dataset_class(self.TRAIN_IMG_ROOT, self.ANN_PICKLE_PATH, self.TRAIN_PICKLE_PATH, self.IMG_SIZE, self.MEAN, self.STD, weak_aug)
        self.val_dataset = dataset_class(self.TRAIN_IMG_ROOT, self.ANN_PICKLE_PATH, self.VAL_PICKLE_PATH, self.IMG_SIZE, self.MEAN, self.STD, val_aug)
        self.test_dataset = dataset_class(self.TEST_IMG_ROOT, self.ANN_PICKLE_PATH, self.TEST_PICKLE_PATH, self.IMG_SIZE, self.MEAN, self.STD, val_aug, 'test')
        
        if distr:
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
            self.test_sampler = DistributedSampler(self.test_dataset,shuffle=False)
            self.dataloader_train = DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE,
                             num_workers=self.N_WORKERS, sampler=self.train_sampler)
            self.dataloader_val = DataLoader(self.val_dataset, batch_size=1,
                                     num_workers=self.N_WORKERS, sampler=self.val_sampler)
            self.dataloader_test = DataLoader(self.test_dataset, batch_size=1,
                                    num_workers=self.N_WORKERS, sampler=self.test_sampler)
        else:
            self.dataloader_train = DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE,
                             num_workers=self.N_WORKERS, shuffle=True)
            self.dataloader_val = DataLoader(self.val_dataset, batch_size=1,
                                     num_workers=self.N_WORKERS, shuffle=False)
            self.dataloader_test = DataLoader(self.test_dataset, batch_size=1,
                                    num_workers=self.N_WORKERS, shuffle=False)
        
        #self.device = torch.device(self.DEVICE if torch.cuda.is_available() else "cpu")
        if distr:
            self.device = torch.device('cuda', args.local_rank) if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(self.DEVICE if torch.cuda.is_available() else "cpu")

        self.model = smp.Unet(self.MODEL_ALIAS, encoder_weights="imagenet", activation='sigmoid')
        self.model.to(self.device)
        
        opt_params = config_dict['optimizer_params']
        opt_params['params'] = self.model.parameters()
        optimizer_class = globals()[config_dict['optimizer']]
        self.optimizer = optimizer_class(**opt_params)
        loss_class = globals()[config_dict['loss']]
        self.criterion = loss_class(**config_dict['loss_params'])
        self.criterion.cuda(None)
        self.dice_coef = DiceCoef(config_dict['smooth'])
        self.dice_comp_coef = DiceCompCoef(config_dict['SIGM_THR'])
        
        self.fold = self.TRAIN_PICKLE_PATH.split('_')[-1][0]
        
        scheduler_class = globals()[config_dict['scheduler']]
        scheduler_params = config_dict['scheduler_params']
        scheduler_params['optimizer'] = self.optimizer
        self.scheduler = scheduler_class(**scheduler_params)
        
        self.snap_fold_name = "_".join([self.MODEL_ALIAS, self.fold, str(self.IMG_SIZE)]+[str(el) for el in time.localtime()[:-3]])
        self.val_metrics = np.array([], dtype=np.float32)
        self.snap_names = np.array([], dtype='U255')

        if use_amp:
             self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O2", keep_batchnorm_fp32=True, loss_scale="dynamic")

        if distr:
            self.model = nn.DataParallel(self.model)

        if config_dict['CUR_SNAPSHOT']:
            if distr:
                self.model.load_state_dict(torch.load(config_dict['CUR_SNAPSHOT']))
            else:
                model_dic = torch.load(config_dict['CUR_SNAPSHOT'])
                new_state_dict = collections.OrderedDict() 
                for k, v in model_dic.items():
                    name = k[7:]
                    new_state_dict[name] = v

                self.model.load_state_dict(new_state_dict)

    def fit(self):
        for epoch in range(self.N_EPOCHS):
            train_loss = 0
            val_loss = 0
            self.scheduler.step()
            train_loss, _ = train_model(self.model, self.dataloader_train, self.optimizer, self.criterion, self.scheduler, epoch, False, self.device)
            train_loss = np.mean(train_loss)
            if epoch == 0:
                writer = SummaryWriter(os.path.join(self.LOGS_PATH, self.snap_fold_name))
                snap_fold_name = os.path.join(self.SNAPSHOTS_PATH, self.snap_fold_name)
                if not os.path.exists(snap_fold_name):
                    if not os.path.exists(snap_fold_name):
                        os.mkdir(snap_fold_name)

            val_loss, dice, dice_comp, fpr = validate_model(self.model, self.dataloader_val, self.criterion, self.dice_coef, self.dice_comp_coef, self.device)
            print("Train loss:", train_loss, " Val loss:", val_loss, "Dice coef:", dice, "Dice comp:", dice_comp,
                 "FPR:", fpr)
            writer.add_scalar('Train/Loss', train_loss, epoch)
            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Val/DiceCoef', dice, epoch)
            writer.add_scalar('Val/DiceCompCoef', dice_comp, epoch)
            writer.add_scalar('Val/FPR', fpr, epoch)
            try:
                if len(self.val_metrics) < 3:
                    self.val_metrics = np.append(self.val_metrics,dice_comp)
                    snap_path = os.path.join(snap_fold_name, self.MODEL_ALIAS+'_'+str(epoch)+'_'+str(np.round(dice_comp, 3))+'.pth')
                    self.snap_names = np.append(self.snap_names,snap_path)
                    torch.save(self.model.state_dict(), snap_path)
                else:
                    if dice_comp > self.val_metrics[0]:
                        self.val_metrics[0] = dice_comp
                        snap_path = os.path.join(snap_fold_name, self.MODEL_ALIAS+'_'+str(epoch)+'_'+str(np.round(dice_comp, 3))+'.pth')
                        os.remove(self.snap_names[0])
                        self.snap_names[0] = snap_path
                        torch.save(self.model.state_dict(), snap_path)
            except:
                pass
            self.snap_names = self.snap_names[np.argsort(self.val_metrics)]
            self.val_metrics = np.sort(self.val_metrics)
