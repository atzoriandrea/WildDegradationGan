import os
import sys
import numpy as np
import cv2
import random
from torchvision import transforms

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from load_data import faces_data, test_set_data, High_Data, Low_Data, Test_Data
from degradation_model import High2Low
from discriminator import Discriminator
from model import GEN_DEEP

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--gpu", action="store", dest="gpu", help="separate numbers with commas, eg. 3,4,5",
                    required=True)


if __name__ == "__main__":
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpus = args.gpu.split(",")
    n_gpu = len(gpus)

    seed_num = 2020
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    max_epoch = 50
    learn_rate = 1e-4
    alpha, beta = 1, 0.05

    G_h2l = High2Low().cuda()
    D_h2l = Discriminator(16).cuda()
    mse = nn.MSELoss()
    optim_D_h2l = optim.Adam(filter(lambda p: p.requires_grad, D_h2l.parameters()), lr=learn_rate, betas=(0.0, 0.9))
    optim_G_h2l = optim.Adam(G_h2l.parameters(), lr=learn_rate, betas=(0.0, 0.9))

    data = faces_data(High_Data, Low_Data)
    test_data = test_set_data(Test_Data)
    loader = DataLoader(dataset=data, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    n_batches = len(loader)
    test_save = "intermid_results"
    for ep in range(1, max_epoch + 1):
        G_h2l.train()
        D_h2l.train()
        for i, batch in enumerate(loader):
            optim_D_h2l.zero_grad()
            optim_G_h2l.zero_grad()

            zs = batch["z"].cuda()
            lrs = batch["lr"].cuda()
            hrs = batch["hr"].cuda()
            downs = batch["hr_down"].cuda()

            lr_gen = G_h2l(hrs)#, zs)
            lr_gen_detach = lr_gen.detach()

            # update discriminator
            loss_D_h2l = nn.ReLU()(1.0 - D_h2l(lrs)).mean() + nn.ReLU()(1 + D_h2l(lr_gen_detach)).mean()
            loss_D_h2l.backward()
            optim_D_h2l.step()

            # update generator
            optim_D_h2l.zero_grad()
            gan_loss_h2l = -D_h2l(lr_gen).mean()
            mse_loss_h2l = mse(lr_gen, downs)

            loss_G_h2l = alpha * mse_loss_h2l + beta * gan_loss_h2l
            loss_G_h2l.backward()
            optim_G_h2l.step()

            print(" %d/%d(%d) D_h2l: %.3f, G_h2l: %.3f \r" % (i + 1, int(n_batches), ep, loss_D_h2l.item(), loss_G_h2l.item()), flush=True)

        print("\n Testing and saving...")
        G_h2l.eval()
        D_h2l.eval()
        for i, sample in enumerate(test_loader):
            high_temp = sample["hr"].numpy()
            high = torch.from_numpy(np.ascontiguousarray(high_temp[:, ::-1, :, :])).cuda()
            with torch.no_grad():
                high_gen = G_h2l(high)
            np_gen = high_gen.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
            np_gen = (np_gen - np_gen.min()) / (np_gen.max() - np_gen.min())
            np_gen = (np_gen * 255).astype(np.uint8)
            cv2.imwrite("{}/imgs/{}_{}_sr.png".format(test_save, ep, i + 1), np_gen)
        save_file = "{}/models/model_epoch_{:03d}.pth".format(test_save, ep)
        torch.save({"G_h2l": G_h2l.state_dict(), "D_h2l": D_h2l.state_dict()}, save_file)
        print("saved: ", save_file)
    print("finished.")