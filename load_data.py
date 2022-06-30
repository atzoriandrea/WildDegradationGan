import os, sys
import numpy as np
import cv2
import itertools
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as nnF
import torch

High_Data = ["/media/Workspace/Datasets/JSTSP/UnpairedSR_Datasets/HIGH/celea_60000_SFD",
             "/media/Workspace/Datasets/JSTSP/UnpairedSR_Datasets/HIGH/SRtrainset_2",
             "/media/Workspace/Datasets/JSTSP/UnpairedSR_Datasets/HIGH/vggface2/vggcrop_test_lp10",
             "/media/Workspace/Datasets/JSTSP/UnpairedSR_Datasets/HIGH/vggface2/vggcrop_train_lp10"]

Low_Data = ["/media/Workspace/Datasets/JSTSP/QMUL-SurvFace-v1/QMUL-SurvFace/SLR_cropped"]

Test_Data = ["/media/Workspace/Datasets/IJCB/DiveFaceResizedFaceX/Dataset/7259735@N07_identity_10",
             "/media/Workspace/Datasets/IJCB/DiveFaceResizedFaceX/Dataset/7597392@N03_identity_7",
             "/media/Workspace/Datasets/IJCB/DiveFaceResizedFaceX/Dataset/8495919@N02_identity_31"]


def get_files_full_path(rootdir):
    import os
    paths = []
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            paths.append(os.path.join(root, file))
    return paths


class faces_data(Dataset):
    def __init__(self, data_hr, data_lr):
        hr = []
        lr = []
        hr_temp = [get_files_full_path(path) for path in data_hr]
        lr_temp = [get_files_full_path(path) for path in data_lr]
        hr, lr = [],[]
        [hr.extend(el) for el in hr_temp]
        [lr.extend(el) for el in lr_temp]
        self.hr_imgs = hr  #[os.path.join(d, i) for d in data_hr for i in os.listdir(d) if os.path.isfile(os.path.join(d, i))]
        self.lr_imgs = lr #[os.path.join(d, i) for d in data_lr for i in os.listdir(d) if os.path.isfile(os.path.join(d, i))]
        self.lr_len = len(self.lr_imgs)
        self.lr_shuf = np.arange(self.lr_len)
        np.random.shuffle(self.lr_shuf)
        self.lr_idx = 0
        self.preproc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.hr_imgs)

    def __getitem__(self, index):
        data = {}
        hr = cv2.imread(self.hr_imgs[index])
        lr = cv2.resize(cv2.imread(self.lr_imgs[self.lr_shuf[self.lr_idx]]),(16,16))
        self.lr_idx += 1
        if self.lr_idx >= self.lr_len:
            self.lr_idx = 0
            np.random.shuffle(self.lr_shuf)
        data["z"] = torch.randn(1, 64, dtype=torch.float32)
        data["lr"] = self.preproc(lr)
        data["hr"] = self.preproc(hr)
        data["hr_down"] = nnF.avg_pool2d(data["hr"], 4, 4)
        return data
    
    def get_noise(self, n):
        return torch.randn(n, 1, 64, dtype=torch.float32)


class test_set_data(Dataset):
    def __init__(self, data_hr):
        hr_temp = [get_files_full_path(path) for path in data_hr]
        hr = []
        [hr.extend(el) for el in hr_temp]
        self.hr_imgs = hr  #[os.path.join(d, i) for d in data_hr for i in os.listdir(d) if os.path.isfile(os.path.join(d, i))]
        self.preproc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.hr_imgs)

    def __getitem__(self, index):
        data = {}
        hr = cv2.resize(cv2.imread(self.hr_imgs[index]),(64,64))
        data["z"] = torch.randn(1, 64, dtype=torch.float32)
        data["hr"] = self.preproc(hr)
        return data

"""
if __name__ == "__main__":
    data = faces_data(High_Data, Low_Data)
    loader = DataLoader(dataset=data, batch_size=16, shuffle=True)
    for i, batch in enumerate(loader):
        print("batch: ", i)
        lrs = batch["lr"].numpy()
        hrs = batch["hr"].numpy()
        downs = batch["hr_down"].numpy()

        for b in range(batch["z"].size(0)):
            lr = lrs[b]
            hr = hrs[b]
            down = downs[b]
            lr = lr.transpose(1, 2, 0)
            hr = hr.transpose(1, 2, 0)
            down = down.transpose(1, 2, 0)
            lr = (lr - lr.min()) / (lr.max() - lr.min())
            hr = (hr - hr.min()) / (hr.max() - hr.min())
            down = (down - down.min()) / (down.max() - down.min())
            cv2.imshow("lr-{}".format(b), lr)
            cv2.imshow("hr-{}".format(b), hr)
            cv2.imshow("down-{}".format(b), down)
            cv2.waitKey()
            cv2.destroyAllWindows()

    print("finished.")
"""