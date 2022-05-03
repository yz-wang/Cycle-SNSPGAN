#!/usr/bin/python3

import argparse
import itertools
import os
from pathlib import Path

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
from models import Generator
from SN_GAN import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset
from utils import VGGNet
from color_loss import Blur
from TV_loss import tv_loss

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=2, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/hazy2clear/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--cn_input_size', type=int, default=256)
parser.add_argument('--ld_input_size', type=int, default=64)
parser.add_argument('--data_parallel', action='store_true')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


def compute_soft_min_max(patch_region_likelihoods, alpha):

    # Compute the log likelihoods for computational stability
    log_likelihood = torch.log(patch_region_likelihoods + 1e-10)
    # Calculating the weights
    exp_log_likelihood = torch.exp(alpha * log_likelihood)
    exp_log_likelihood_sum = torch.sum(exp_log_likelihood, dim=3)
    # Summing all the weights for each image -> batch_size x 1
    exp_log_likelihood_sum = torch.sum(exp_log_likelihood_sum, dim=2)
    weights = exp_log_likelihood / exp_log_likelihood_sum.unsqueeze(1).unsqueeze(1)
    # weight the log likelihoods with the weights
    soft_min_val = torch.sum(log_likelihood * weights, dim=3)
    soft_min_val = torch.sum(soft_min_val, dim=2)

    return torch.exp(soft_min_val)


def patchGan(output, alpha, min_max_patchGan_func):
    if patchGan_func:
        output = compute_soft_min_max(output, alpha).squeeze(-1)
    else:
        output = torch.mean(output, dim=3).mean(dim=2).squeeze(1)

    return output


def update_patch_function(epoch, min=False):
    if epoch > 28:
        min_func = True
        return min_func

    else:
        return False


###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)

netD_A = Discriminator(opt.input_nc, 64)
netD_B = Discriminator(opt.output_nc, 64)

if opt.cuda:
    netG_A2B.cuda(0)
    netG_B2A.cuda(0)
    netD_A.cuda(0)
    netD_B.cuda(0)

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# pretrained VGG16 module set in evaluation mode for feature extraction
vgg = VGGNet().cuda().eval()


def perceptual_loss(x, y):
    # c = torch.nn.MSELoss()
    c = torch.nn.L1Loss()
    # rx = netG_B2A(netG_A2B(x))  # reconstructA
    # ry = netG_A2B(netG_B2A(y))  # reconstructB
    fb = netG_A2B(x).squeeze(-1)  # FakeB
    fa = netG_B2A(y).squeeze(-1)  # FakeA

    fx1, fx2 = vgg(x)
    fy1, fy2 = vgg(y)

    frx1, frx2 = vgg(fb)
    fry1, fry2 = vgg(fa)

    m1 = c(fx1, frx1)
    m2 = c(fx2, frx2)

    m3 = c(fy1, fry1)
    m4 = c(fy2, fry2)

    loss = (m1 + m2 + m3 + m4) * 0.5 * 5

    return loss


# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
blur = Blur(3).cuda(0)
cl = torch.nn.MSELoss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                               lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)  # 全填充为1
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)  # 全填充为0

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [transforms.Resize(int(opt.size*1.12), Image.BICUBIC),
               transforms.RandomCrop(opt.size),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

if os.path.exists('output/netG_A2B.pth'):
    checkpoint = torch.load('output/netG_A2B.pth')
    netG_A2B.load_state_dict(checkpoint['model'])
    optimizer_G.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print('Load netG_A2B epoch {} success！'.format(start_epoch))
else:
    start_epoch = 0
    print('No saved netG_A2B model, will start training from scratch')

if os.path.exists('output/netG_B2A.pth'):
    checkpoint = torch.load('output/netG_B2A.pth')
    netG_B2A.load_state_dict(checkpoint['model'])
    optimizer_G.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print('Load netG_B2A epoch {} success！！'.format(start_epoch))
else:
    start_epoch = 0
    print('No saved netG_B2A model，will start training from scratch！')

if os.path.exists('output/netD_A.pth'):
    checkpoint = torch.load('output/netD_A.pth')
    netD_A.load_state_dict(checkpoint['model'])
    optimizer_D_A.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print('Load netD_A epoch {} success！'.format(start_epoch))
else:
    start_epoch = 0
    print('No saved netD_A model，will start training from scratch！')

if os.path.exists('output/netD_B.pth'):
    checkpoint = torch.load('output/netD_B.pth')
    netD_B.load_state_dict(checkpoint['model'])
    optimizer_D_B.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print('Load netD_B epoch {} success！'.format(start_epoch))
else:
    start_epoch = 0
    print('No saved netD_B model，will start training from scratch！')

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader), start_epoch)

initial_epoch = start_epoch
patchGan_func = update_patch_function(initial_epoch)
###################################

###### Training ######
for epoch in range(start_epoch, opt.n_epochs):
    nan_flag = False
    for i, batch in enumerate(dataloader):
        if update_patch_function(epoch):
            patchGan_func = update_patch_function(epoch)
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * 5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B).squeeze(-1)
        alpha = -1
        pred_fake = patchGan(pred_fake, alpha, patchGan_func)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A).squeeze(-1)
        alpha = -1
        pred_fake = patchGan(pred_fake, alpha, patchGan_func)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

        # Perceptual loss
        loss_perceptual = perceptual_loss(real_A, real_B)

        # Color loss
        loss_color = cl(blur(fake_B).squeeze(-1), blur(real_B).squeeze(-1)) * 0.05

        # TV loss
        loss_TV = (tv_loss(fake_B) + tv_loss(fake_A)) * 0.5
        # Total loss
        loss_G = loss_color + loss_perceptual + loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_TV
        if torch.isnan(loss_G):
            nan_flag = True
            break
        loss_G.backward()
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A).squeeze(-1)
        alpha = -1
        pred_real = patchGan(pred_real, alpha, patchGan_func)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach()).squeeze(-1)
        alpha = 1
        pred_fake = patchGan(pred_fake, alpha, patchGan_func)
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss

        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        if torch.isnan(loss_D_A):
            nan_flag = True
            break
        loss_D_A.backward()
        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B).squeeze(-1)
        alpha = -1
        pred_real = patchGan(pred_real, alpha, patchGan_func)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach()).squeeze(-1)
        alpha = 1
        pred_fake = patchGan(pred_fake, alpha, patchGan_func)
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()
        optimizer_D_B.step()
        ###################################

        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B),
                    'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B),
                    'loss_G_perceptual': loss_perceptual, 'loss_color': loss_color, 'loss_TV': loss_TV},
                   images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

    if nan_flag:
        print("training process stop because of nan problems")
        break

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    save_folder = Path("./output")


    state = {'model': netG_A2B.state_dict(), 'optimizer': optimizer_G.state_dict(), 'epoch': epoch}
    torch.save(state, save_folder / "netG_A2B.pth")
    state = {'model': netG_B2A.state_dict(), 'optimizer': optimizer_G.state_dict(), 'epoch': epoch}
    torch.save(state, save_folder / 'netG_B2A.pth')
    state = {'model': netD_A.state_dict(), 'optimizer': optimizer_D_A.state_dict(), 'epoch': epoch}
    torch.save(state, save_folder / 'netD_A.pth')
    state = {'model': netD_B.state_dict(), 'optimizer': optimizer_D_B.state_dict(), 'epoch': epoch}
    torch.save(state, save_folder / 'netD_B.pth')

    # python train.py --dataroot datasets/hazy2clear/ --cuda
    # python test.py --dataroot datasets/hazy2clear/ --cuda

