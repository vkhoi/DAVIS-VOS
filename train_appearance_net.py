import argparse
import os
import sys
import torch
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloaders.davis_2016 import Davis2016
from dataloaders import custom_transforms as tr
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

def evaluate(net, dataloader, cuda=False):
    total = 0
    avg_loss = 0

    for i, samples in enumerate(dataloader):
        gts = Variable(samples["gt"])
        images = Variable(samples["image"])

        if cuda:
            gts = gts.cuda()
            images = images.cuda()
        
        outputs = net(images)
        
        loss = class_balanced_cross_entropy_loss(outputs, gts)

        avg_loss += loss.data[0]
        total += gts.size()[0]

    avg_loss /= total

    return avg_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Appearance Net")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="batch size for training (default: 1)")
    parser.add_argument("--n_epochs", type=int, default=3,
                        help="number of epochs to train (default: 3)")
    parser.add_argument("--resume_epoch", type=int, default=0,
                        help="resume from what epoch (default: 0)")
    parser.add_argument("--log", type=int, default=10,
                        help="log frequency (default: 10 iterations)")
    parser.add_argument("--cuda", type=int, default=0,
                        help="whether to use cuda (default: 0)")
    args = parser.parse_args()

    batch_size = args.batch_size
    n_epochs = args.n_epochs
    resume_epoch = args.resume_epoch
    log_interval = args.log

    n_avg_grad = 10

    if args.cuda == 1:
        print("cuda flag is turned on!")
        cuda = True
    else:
        cuda = False

    print("Loading train set...", end="")
    train_set = Davis2016(transform=transforms.Compose(
        [tr.RandomHorizontalFlip(),
         tr.RandomColorIntensity(),
         tr.ToTensor()]))
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                              shuffle=True)
    train_loader_seq = DataLoader(dataset=train_set, batch_size=batch_size,
                                  shuffle=False)
    print("Done!")

    if resume_epoch == 0:
        net = AppearanceModel()
        optimizer = setup_optimizer(net)
        hist = []
    else:
        net = AppearanceModel(pretrained=False)
        optimizer = setup_optimizer(net)

        print("Loading weights from previous checkpoint...", end="")
        sys.stdout.flush()
        checkpoint = torch.load(os.path.join(
            CHKPT_PATH, MODEL_NAME + ("_epoch-%d" % resume_epoch) + ".chkpt"))

        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        hist = checkpoint["hist"]

        print("Done!")
    
    if cuda:
        net.cuda()

    iters_avg_grad = 0

    print("Start training")
    for epoch in range(resume_epoch + 1, resume_epoch + n_epochs + 1):
        for i, samples in enumerate(train_loader):
            gts = Variable(samples["gt"])
            images = Variable(samples["image"])

            if cuda:
                gts = gts.cuda()
                images = images.cuda()
            
            outputs = net(images)
            
            loss = class_balanced_cross_entropy_loss(outputs, gts)

            if i % log_interval == log_interval - 1:
                print("Epoch %d/%d, iteration %d/%d, loss: %s" % (epoch, 
                      resume_epoch + n_epochs, i + 1,
                      (len(train_set) - 1) // batch_size + 1, loss.data[0]))

            loss /= n_avg_grad
            loss.backward()
            iters_avg_grad += 1
            
            if iters_avg_grad % n_avg_grad == 0:
                optimizer.step()
                optimizer.zero_grad()
                iters_avg_grad = 0

        print("Evaluating on train set...: ", end="")
        sys.stdout.flush()
        train_loss = evaluate(net, train_loader_seq, cuda)
        print("train_loss = %.9f" % train_loss)
        hist.append({
            "train_loss": train_loss
        })

        checkpoint = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "hist": hist
        }

        # Save checkpoint.
        torch.save(checkpoint, os.path.join(
            CHKPT_PATH, MODEL_NAME + ("_epoch-%d" % epoch) + ".chkpt"))
