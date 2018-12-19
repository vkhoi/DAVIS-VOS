import argparse
import copy
import cv2
import davis
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim

import dataloaders.custom_transforms as tr

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloaders.davis_2016 import Davis2016
from helper import class_balanced_cross_entropy_loss
from models.appearance_flow_model import AppearanceFlowModel

CHKPT_PATH = "models/trained"
MODEL_NAME = "AppearanceFlowModel"

def setup_optimizer(net):
    lr_app = 1e-8
    lr_flow = 1e-8
    lr_fuse = 1e-5
    wd = 0.0002
    optimizer = optim.SGD([
        {'params': [param[1] for param in net.app_blocks.named_parameters()
                    if 'weight' in param[0]],
         'weight_decay': wd,
         'lr': lr_app },
        {'params': [param[1] for param in net.app_blocks.named_parameters()
                    if 'bias' in param[0]],
         'lr': 2 * lr_app },
        {'params': [param[1] for param in net.app_sides.named_parameters()
                    if 'weight' in param[0]],
         'weight_decay': wd,
         'lr': lr_app },
        {'params': [param[1] for param in net.app_sides.named_parameters()
                    if 'bias' in param[0]],
         'lr': 2 * lr_app },
        ############
        {'params': [param[1] for param in net.flow_blocks.named_parameters()
                    if 'weight' in param[0]],
         'weight_decay': wd,
         'lr': lr_flow },
        {'params': [param[1] for param in net.flow_blocks.named_parameters()
                    if 'bias' in param[0]],
         'lr': 2 * lr_flow },
        {'params': [param[1] for param in net.flow_sides.named_parameters()
                    if 'weight' in param[0]],
         'weight_decay': wd,
         'lr': lr_flow },
        {'params': [param[1] for param in net.flow_sides.named_parameters()
                    if 'bias' in param[0]],
         'lr': 2 * lr_flow },
        ############
        {'params': net.fuse.weight,
         'lr': lr_fuse / 100,
         'weight_decay': wd },
        {'params': net.fuse.bias,
         'lr': 2 * lr_fuse / 100 },
    ], lr=lr_app, momentum=0.9)

    return optimizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Appearance Net")
    parser.add_argument("--result_dir", type=str, default="results",
                        help="directory to write results")
    parser.add_argument("--net", type=str, default="ApperanceFlowModel_epoch-20",
                        help="filename of network checkpoint")
    parser.add_argument("--fname", type=str, default="blackswan",
                        help="name of video in the validation set")
    parser.add_argument("--n_iterations", type=int, default=200,
                        help="number of iterations to finetune (default: 200)")
    parser.add_argument("--log", type=int, default=10,
                        help="log frequency (default: 10 iterations)")
    parser.add_argument("--cuda", type=int, default=0,
                        help="whether to use cuda (default: 0)")
    parser.add_argument("--maskprop", type=int, default=0)
    args = parser.parse_args()

    result_dir = args.result_dir
    net_name = args.net
    fname = args.fname
    n_iterations = args.n_iterations
    log_interval = args.log

    use_mask_prop = False
    if args.maskprop == 1:
        use_mask_prop = True
        print("maskprop flag is turned on!")

    if args.cuda == 1:
        print("cuda flag is turned on!")
        cuda = True
    else:
        cuda = False

    print("Loading images of %s..." % fname, end="")
    dataset = Davis2016(train=False, fname=fname,   
                        transform=transforms.Compose([tr.RandomHorizontalFlip(),
                                                      tr.RandomColorIntensity(),
                                                      tr.ToTensor()]),
                        mode="app_flow-intensity")
    print("Done!")

    net = AppearanceFlowModel(pretrained=False)
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
    sys.stdout.flush()

    for i in range(n_iterations):
        sample = dataset.__getitem__(0)
        gt = Variable(torch.unsqueeze(sample["gt"], 0))
        image = Variable(torch.unsqueeze(sample["image"], 0))
        flow = Variable(torch.unsqueeze(sample["flow_intensity"], 0))
        if use_mask_prop:
            # flow_raw = Variable(sample["flow_raw"])
            # gt_prev = Variable(sample["gt_prev"].unsqueeze(0))
            mask = gt.clone()

        if cuda:
            gt = gt.cuda()
            image = image.cuda()
            flow = flow.cuda()
            if use_mask_prop:
                mask = mask.cuda()

        if use_mask_prop:
            output = net(image, flow, mask)
        else:
            output = net(image, flow)

        loss = class_balanced_cross_entropy_loss(output, gt)
        if i % log_interval == log_interval - 1:
            print("Iteration %d/%d, loss: %s"
                  % (i + 1, n_iterations, loss.data[0]))
            sys.stdout.flush()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.save(net.state_dict(), os.path.join(
        CHKPT_PATH, net_name + "_finetuned-" + fname + ".chkpt"))

    print("Running on validation set")
    result_folder = os.path.join(result_dir, fname)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    dataset = Davis2016(train=False, fname=fname,
                        transform=transforms.Compose([tr.ToTensor()]),
                        mode="app_flow-intensity")

    mask_prev = None

    for i in range(len(dataset)):
        sample = dataset.__getitem__(i)
        gt = Variable(torch.unsqueeze(sample["gt"], 0))
        image = Variable(torch.unsqueeze(sample["image"], 0))
        flow = Variable(torch.unsqueeze(sample["flow_intensity"], 0))

        if i == 0:
            mask_prev = gt
            gt = gt.data.numpy().squeeze()
            gt[gt > 0] = 1
            gt = gt.astype(np.uint8)
            davis.io.imwrite_indexed(
                os.path.join(result_folder, "%05d.png" % i), gt)
            # gt = (gt * 255).astype(np.uint8)
            # cv2.imwrite(os.path.join(result_folder, "%05d.png" % i), gt)
        else:
            if cuda:
                image = image.cuda()
                flow = flow.cuda()
                if use_mask_prop:
                    mask_prev = mask_prev.cuda()
                    output = net(image, flow, mask_prev).data.cpu().numpy()
                else:
                    output = net(image, flow).data.cpu().numpy().squeeze()
            else:
                if use_mask_prop:
                    output = net(image, flow, mask_prev).data.numpy()
                else:
                    output = net(image, flow).data.numpy().squeeze()

            output[output >= 0] = 1
            output[output < 0] = 0
            if use_mask_prop:
                mask_prev = copy.deepcopy(output)
                mask_prev = Variable(torch.from_numpy(mask_prev))
                output = output.squeeze()

            output = (output).astype(np.uint8)
            davis.io.imwrite_indexed(
                os.path.join(result_folder, "%05d.png" % i), output)
            # cv2.imwrite(os.path.join(result_folder, "%05d.png" % i), output)
            
