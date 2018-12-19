import argparse
import cv2
import davis
import numpy as np
import os
import sys
import torch
import torch.optim as optim

import dataloaders.custom_transforms as tr

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloaders.davis_2016 import Davis2016
from models.helper import class_balanced_cross_entropy_loss
from models.appearance_model import AppearanceModel

CHKPT_PATH = "models/trained"
MODEL_NAME = "AppearanceModel"

def setup_optimizer(net):
    lr = 1e-8
    wd = 0.0002
    optimizer = optim.SGD([
        {'params': [param[1] for param in net.blocks.named_parameters()
                    if 'weight' in param[0]],
         'weight_decay': wd,
         'lr': lr },
        {'params': [param[1] for param in net.blocks.named_parameters()
                    if 'bias' in param[0]],
         'lr': 2 * lr },
        {'params': [param[1] for param in net.sides.named_parameters()
                    if 'weight' in param[0]],
         'weight_decay': wd,
         'lr': lr },
        {'params': [param[1] for param in net.sides.named_parameters()
                    if 'bias' in param[0]],
         'lr': 2 * lr },
        {'params': [param[1] for param in net.upsamples.named_parameters()
                    if 'weight' in param[0]],
         'lr': 0 },
        {'params': net.fuse.weight,
         'lr': lr / 100,
         'weight_decay': wd },
        {'params': net.fuse.bias,
         'lr': 2 * lr / 100 },
    ], lr=lr, momentum=0.9)

    return optimizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Appearance Net")
    parser.add_argument("--result_dir", type=str, default="results",
                        help="directory to write results")
    parser.add_argument("--net", type=str, default="VOSModel_epoch-20",
                        help="filename of network checkpoint")
    parser.add_argument("--fname", type=str, default="blackswan",
                        help="name of video in the validation set")
    parser.add_argument("--n_iterations", type=int, default=200,
                        help="number of iterations to finetune (default: 200)")
    parser.add_argument("--log", type=int, default=10,
                        help="log frequency (default: 10 iterations)")
    parser.add_argument("--cuda", type=int, default=0,
                        help="whether to use cuda (default: 0)")
    args = parser.parse_args()

    result_dir = args.result_dir
    net_name = args.net
    fname = args.fname
    n_iterations = args.n_iterations
    log_interval = args.log

    if args.cuda == 1:
        print("cuda flag is turned on!")
        cuda = True
    else:
        cuda = False

    print("Loading images of %s..." % fname, end="")
    dataset = Davis2016(train=False, fname=fname,   
                        transform=transforms.Compose([tr.RandomHorizontalFlip(),
                                                      tr.RandomColorIntensity(),
                                                      tr.ToTensor()]))
    print("Done!")

    net = AppearanceModel(pretrained=False)
    optimizer = setup_optimizer(net)

    print("Loading network %s..." % net_name, end="")
    sys.stdout.flush()
    if cuda:
        checkpoint = torch.load(os.path.join(CHKPT_PATH, net_name + ".chkpt"))
    else:
        checkpoint = torch.load(os.path.join(CHKPT_PATH, net_name + ".chkpt"),
                                map_location=lambda storage, loc: storage)

    net.load_state_dict(checkpoint["net"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("Done!")

    if cuda:
        net = net.cuda()

    print("Start finetuning")
    for i in range(n_iterations):
        break
        sample = dataset.__getitem__(0)
        gt = Variable(torch.unsqueeze(sample["gt"], 0))
        image = Variable(torch.unsqueeze(sample["image"], 0))

        if cuda:
            gt = gt.cuda()
            image = image.cuda()

        output = net(image)

        loss = class_balanced_cross_entropy_loss(output, gt)
        if i % log_interval == log_interval - 1:
            print("Iteration %d/%d, loss: %s"
                  % (i + 1, n_iterations, loss.data[0]))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print("Running on validation set")
    result_folder = os.path.join(result_dir, fname)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    dataset = Davis2016(train=False, fname=fname,
                        transform=transforms.Compose([tr.ToTensor()]))
    for i in range(len(dataset)):
        sample = dataset.__getitem__(i)
        gt = Variable(torch.unsqueeze(sample["gt"], 0))
        image = Variable(torch.unsqueeze(sample["image"], 0))

        if i == 0:
            gt = gt.data.numpy().squeeze().astype(np.uint8)
            davis.io.imwrite_indexed(
                os.path.join(result_folder, "%05d.png" % i), gt)
            # gt = (gt * 255).astype(np.uint8)
            # cv2.imwrite(os.path.join(result_folder, "%05d.png" % i), gt)
        else:
            if cuda:
                gt = gt.cuda()
                image = image.cuda()
                output = net(image).data.cpu().numpy().squeeze()
            else:
                output = net(image).data.numpy().squeeze()

            output[output >= 0] = 1
            output[output < 0] = 0
            output = (output).astype(np.uint8)
            davis.io.imwrite_indexed(
                os.path.join(result_folder, "%05d.png" % i), output)
            # cv2.imwrite(os.path.join(result_folder, "%05d.png" % i), output)
