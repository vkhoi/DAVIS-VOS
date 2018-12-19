import argparse
import copy
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloaders.davis_2016 import Davis2016
from dataloaders.davis_2017 import Davis2017
from dataloaders import custom_transforms as tr
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

def evaluate(net, dataloader, cuda=False, use_mask_prev=False):
    total = 0
    avg_loss = 0

    for i, samples in enumerate(dataloader):
        gts = Variable(samples["gt"])
        images = Variable(samples["image"])
        flows = Variable(samples["flow_intensity"])

        if use_mask_prev:
            mask = Variable(samples["gt_prev"].unsqueeze(0))

        if cuda:
            gts = gts.cuda()
            images = images.cuda()
            flows = flows.cuda()
            if use_mask_prev:
                mask = mask.cuda()
            
        if use_mask_prev:
            outputs = net(images, flows, mask)
        else:
            outputs = net(images, flows)

        loss = class_balanced_cross_entropy_loss(outputs, gts)

        avg_loss += loss.data[0]
        total += gts.size()[0]

    avg_loss /= total

    return avg_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Appearance Net")
    parser.add_argument("--net", type=str, default="none",
                        help="filename of network checkpoint")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="batch size for training (default: 1)")
    parser.add_argument("--resume_epoch", type=int, default=0,
                        help="epoch number that we resume from (default: 0)")
    parser.add_argument("--n_epochs", type=int, default=3,
                        help="number of epochs to train (default: 3)")
    parser.add_argument("--log", type=int, default=10,
                        help="log frequency (default: 10 iterations)")
    parser.add_argument("--cuda", type=int, default=0,
                        help="whether to use cuda (default: 0)")
    parser.add_argument("--year", type=int, default=2016)
    parser.add_argument("--maskprev", type=int, default=0)
    parser.add_argument("--trainmask", type=int, default=0)
    parser.add_argument("--len_trainmask", type=int, default=1)
    args = parser.parse_args()

    net_name = args.net
    batch_size = args.batch_size
    resume_epoch = args.resume_epoch
    n_epochs = args.n_epochs
    log_interval = args.log
    year = args.year
    len_trainmask = args.len_trainmask

    use_mask_prev = False
    if args.maskprev == 1:
        use_mask_prev = True
        print("maskprev flag is turned on!")

    train_mask = False
    if args.trainmask == 1:
        train_mask = True
        print("trainmask flag is turned on!")

    n_avg_grad = 10

    if args.cuda == 1:
        print("cuda flag is turned on!")
        cuda = True
    else:
        cuda = False

    if net_name == "none":
        net = AppearanceFlowModel()
        optimizer = setup_optimizer(net)
        hist = []
    else:
        net = AppearanceFlowModel(pretrained=False)
        optimizer = setup_optimizer(net)

        print("Loading weights from previous checkpoint...", end="")
        sys.stdout.flush()
        checkpoint = torch.load(os.path.join(
            CHKPT_PATH, net_name + ".chkpt"))

        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        hist = checkpoint["hist"]

        print("Done!")
    
    if cuda:
        net.cuda()

    iters_avg_grad = 0

    if not train_mask:
        print("Loading train set...", end="")
        if args.year == 2016:
            train_set = Davis2016(
                transform=transforms.Compose([tr.RandomHorizontalFlip(),
                                              tr.RandomColorIntensity(),
                                              tr.ToTensor()]),
                mode="app_flow-intensity")
        else:
            train_set = Davis2017(
                transform=transforms.Compose([tr.RandomHorizontalFlip(),
                                              tr.RandomColorIntensity(),
                                              tr.ToTensor()]),
                mode="app_flow-intensity")
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                                  shuffle=True)
        # train_loader_seq = DataLoader(dataset=train_set, batch_size=batch_size,
        #                               shuffle=False)
        print("Done!")

        print("Start training")
        sys.stdout.flush()
        for epoch in range(resume_epoch + 1, resume_epoch + n_epochs + 1):
            avg_loss = 0

            for i, samples in enumerate(train_loader):
                gts = Variable(samples["gt"])
                images = Variable(samples["image"])
                flows = Variable(samples["flow_intensity"])

                if use_mask_prev:
                    mask = Variable(samples["gt_prev"].unsqueeze(0))

                if cuda:
                    gts = gts.cuda()
                    images = images.cuda()
                    flows = flows.cuda()
                    if use_mask_prev:
                        mask = mask.cuda()
                
                if use_mask_prev:
                    outputs = net(images, flows, mask)
                else:
                    outputs = net(images, flows)
                
                loss = class_balanced_cross_entropy_loss(outputs, gts)
                avg_loss += loss.data[0]

                if i % log_interval == log_interval - 1:
                    print("Epoch %d/%d, iteration %d/%d, loss: %s" % (epoch, 
                          resume_epoch + n_epochs, i + 1,
                          (len(train_set) - 1) // batch_size + 1, loss.data[0]))
                    sys.stdout.flush()

                loss /= n_avg_grad
                loss.backward()
                iters_avg_grad += 1
                
                if iters_avg_grad % n_avg_grad == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    iters_avg_grad = 0

            # print("Evaluating on train set...: ", end="")
            sys.stdout.flush()
            # train_loss = evaluate(net, train_loader_seq, cuda, use_mask_prop)
            print("Average loss on train set = %.9f" % (avg_loss / len(train_loader)))
            hist.append({
                "train_loss": avg_loss
            })

            checkpoint = {
                "net": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "hist": hist
            }

            # Save checkpoint.
            torch.save(checkpoint, os.path.join(
                CHKPT_PATH, MODEL_NAME + ("_epoch-%d" % epoch) + ".chkpt"))
    else:
        print("Loading train set...", end="")
        train_set = Davis2016(
            transform=transforms.Compose([tr.RandomColorIntensity(),
                                          tr.ToTensor()]),
            mode="app_flow-intensity")
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                                  shuffle=True)
        print("Done!")

        N = len(train_loader)
        L = len_trainmask
        n_iters = N // L + 1

        print("Start training mask propagation")
        sys.stdout.flush()
        for epoch in range(resume_epoch + 1, resume_epoch + n_epochs + 1):
            avg_loss = 0

            for i in range(n_iters):
                # Randomly sample a short clip to train.
                while True:
                    start = np.random.randint(N)
                    if start + L >= N:
                        continue

                    ok = True
                    for i_frame in range(start, start + L):
                        if train_set.is_first_frame(i_frame):
                            ok = False
                            break
                    if not ok:
                        continue
                    break
                ######################################
                mask_prev = None
    
                for i_frame in range(start, start + L):
                    sample = train_set.__getitem__(i_frame)
                    image = Variable(sample["image"]).unsqueeze(0)
                    gt = Variable(sample["gt"]).unsqueeze(0)
                    flow = Variable(sample["flow_intensity"]).unsqueeze(0)
                    if mask_prev is None:
                        mask_prev = Variable(sample["gt_prev"]).unsqueeze(0).unsqueeze(1)
                        
                    if cuda:
                        image = image.cuda()
                        gt = gt.cuda()
                        flow = flow.cuda()
                        mask_prev = mask_prev.cuda()

                    out = net(image, flow, mask_prev)
                    loss = class_balanced_cross_entropy_loss(out, gt)
                    avg_loss += loss.data[0]

                    if (i * L + i_frame - start) % log_interval == log_interval - 1:
                        print("Epoch %d/%d, iteration %d/%d, loss: %s" %
                              (epoch, resume_epoch + n_epochs, (i * L + i_frame - start) + 1,
                              n_iters * L, loss.data[0]))
                        sys.stdout.flush()
                    
                    loss /= n_avg_grad
                    loss.backward()
                    iters_avg_grad += 1

                    if iters_avg_grad % n_avg_grad == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        iters_avg_grad = 0
                    
                    if cuda:
                        tmp = out.data.cpu().numpy().squeeze()
                    else:
                        tmp = out.data.numpy().squeeze()
                    mask_prev = copy.deepcopy(tmp)
                    mask_prev[mask_prev >= 0] = 1
                    mask_prev[mask_prev < 0] = 0
                    mask_prev = Variable(torch.from_numpy(mask_prev)).unsqueeze(0).unsqueeze(1)

            print("Average loss on train set = %.9f" % (avg_loss/n_iters/L))
            hist.append({
                "train_loss": avg_loss
            })

            checkpoint = {
                "net": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "hist": hist
            }

            # Save checkpoint.
            torch.save(checkpoint, os.path.join(
                CHKPT_PATH, MODEL_NAME + ("_epoch-%d" % epoch) + \
                ("_mask-%d" % len_trainmask) + ".chkpt"))

