import numpy as np
import random
import torch
import os
import cv2

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6

def top_k_map(img_numpy,k):
    flat_img_numpy = img_numpy.flatten()
    top_2000_indices = np.argpartition(flat_img_numpy, -k)[-k:]
    numpy_new = np.zeros_like(img_numpy)
    numpy_new[np.unravel_index(top_2000_indices, img_numpy.shape)] = 1
    return numpy_new

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def see_img(data,dir,i,type):  #B,C,H,W
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    data = data.cpu()
    data *= std
    data += mean
    data = data*255
    data=data.permute(0,2,3,1)
    data = data[0, :, :, :]
    data = data.cpu().numpy()
    data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    data=data.astype('uint8')
    cv2.imwrite(dir+'/'+f'{type}{i}.png',data)


def see_img_heatmap(data,segresult,dir,i,type,saveorig=False):  #B,C,H,W
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    data = data.cpu()
    data *= std
    data += mean
    data = data*255
    y2max = 255
    y2min = 0
    x2max = segresult.max()
    x2min = segresult.min()
    segresult = np.round((y2max - y2min) * (segresult - x2min) / (x2max - x2min) + y2min)
    segresult = segresult.astype(np.uint8)
    heatmap = cv2.applyColorMap(segresult, colormap=cv2.COLORMAP_JET)
    alpha = 0.15
    alpha2 = 0.3
    data=data.permute(0,2,3,1)
    data = data[0, :, :, :]
    data = data.cpu().numpy()
    data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    data=data.astype('uint8')
    if saveorig:
        cv2.imwrite(dir+'/'+f'{type}{i}_oriimg.png',data)
    overlay = data.copy()
    data = cv2.addWeighted(heatmap, alpha2, overlay, 1 - alpha, 0, overlay)
    cv2.imwrite(dir+'/'+f'{type}{i}.png',data)

def see_img_heatmap_onlyseg(segresult,dir,i,type):  #B,C,H,W
    y2max = 255
    y2min = 0
    x2max = segresult.max()
    x2min = segresult.min()
    segresult = np.round((y2max - y2min) * (segresult - x2min) / (x2max - x2min) + y2min)
    segresult = segresult.astype(np.uint8)
    cv2.imwrite(dir + '/' + f'{type}{i}_outputmask.png', segresult)

def remove_overlap(input_tensor):
    areas = input_tensor.sum(dim=(0, 1))
    sorted_indices = torch.argsort(areas)
    sorted_input_tensor = input_tensor[:, :, sorted_indices]

    output_tensor = torch.zeros_like(input_tensor).cuda()
    mask = torch.zeros(input_tensor.shape[0], input_tensor.shape[1]).bool().cuda()
    for i in range(sorted_input_tensor.shape[-1]):
        current_slice = sorted_input_tensor[:, :, i]
        current_slice[mask] = 0
        output_tensor[:, :, i] = current_slice
        mask |= (current_slice == 1)
    return output_tensor

class mIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        foreground = np.nanmean(iu[1:])
        bg = iu[0]
        FB_IOU = (foreground+bg)/2
        return np.nanmean(iu[1:]), FB_IOU

    def evaluate_withrecall(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        foreground = np.nanmean(iu[1:])
        bg = iu[0]
        FB_IOU = (foreground+bg)/2

        confusion_matrix = self.hist
        num_classes = confusion_matrix.shape[0]
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)

        for i in range(num_classes):
            TP = confusion_matrix[i, i]  # True Positive
            FP = np.sum(confusion_matrix[:, i]) - TP  # False Positive
            FN = np.sum(confusion_matrix[i, :]) - TP  # False Negative

            precision[i] = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall[i] = TP / (TP + FN) if (TP + FN) > 0 else 0

        mean_precision = np.mean(precision)
        mean_recall = np.mean(recall)
        return np.nanmean(iu[1:]), FB_IOU, mean_precision, mean_recall
