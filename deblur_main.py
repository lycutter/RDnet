import torch
import argparse
import numpy as np
from torch.nn import init
import torch.nn as nn
from model.discriminator import Discriminator
from model import generator
from data.data_loader import CreateDataLoader
from torchvision import transforms
from torch.autograd import Variable
from perceptual import PerceptualLoss

parser = argparse.ArgumentParser(description='Semantic aware super-resolution')

parser.add_argument('--model_name', default='RDN', help='model to select')
parser.add_argument('--nDenselayer', type=int, default=3, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
parser.add_argument('--nFeat', type=int, default=64, help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=3, help='number of color channels to use')
parser.add_argument('--nThreads', type=int, default=0, help='number of threads for data loading')
parser.add_argument('--batchSize', type=int, default=4, help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
parser.add_argument('--lossType', default='L1', help='output SR video')
parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--loadSizeX', type=int, default=640, help='scale images to this size')
parser.add_argument('--loadSizeY', type=int, default=360, help='scale images to this size')
parser.add_argument('--dataset_mode', type=str, default='aligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--scale', type=int, default= 4, help='scale output size /input size')
parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')


args = parser.parse_args()

FilePath = './pictureShow/loss.txt'

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data)

def train(train_gen):

    if args.model_name == 'RDN':
        netG = generator.RDN(args).cuda()
    netG.apply(weights_init_kaiming)
    netD = Discriminator().cuda()
    optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, netG.parameters()), 0.0001, [0.9, 0.999])
    optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()), 0.0004, [0.9, 0.999])


    train_gen = train_gen.load_data()
    for epoch in range(1000):
        for i, batch in enumerate(train_gen):
            blur_real = batch['A']
            deblur_real = batch['B']

            blur_real = blur_real.to(0)
            deblur_real = deblur_real.to(0)

            # blur_save = transforms.ToPILImage()(blur.cpu()[0])
            # blur_save.save('./pictureShow/blur_save.jpg')
            # deblur_save = transforms.ToPILImage()(deblur.cpu()[0])
            # deblur_save.save('./pictureShow/deblur_save.jpg')

            deblur_fake = netG(blur_real)
            d_loss_fake = netD(deblur_fake).mean()
            d_loss_real = netD(deblur_real).mean()


            # Compute gradient penalty of HR and sr
            alpha = torch.rand(deblur_real.size(0), 1, 1, 1).cuda().expand_as(deblur_real)
            interpolated = Variable(alpha * deblur_real.data + (1 - alpha) * deblur_fake.data, requires_grad=True)
            disc_interpolates = netD(interpolated)

            grad = torch.autograd.grad(outputs=disc_interpolates,
                                       inputs=interpolated,
                                       grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]

            grad = grad.view(grad.size(0), -1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

            gradient_penalty = 10 * d_loss_gp

            loss_D = d_loss_fake - d_loss_real + gradient_penalty

            optimizer_D.zero_grad()
            loss_D.backward(retain_graph=True)
            optimizer_D.step()

            # calculate loss_G
            deblur_fake = netG(blur_real)
            loss_G_GAN = - torch.mean(deblur_fake)
            image_loss = criterion(deblur_fake, deblur_real)
            content_loss = PerceptualLoss()
            content_loss.initialize(nn.MSELoss())
            contentloss = content_loss.get_loss(deblur_fake, deblur_real)
            loss_G = (contentloss + image_loss) * 100 + loss_G_GAN
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            if i % 50 == 0:
                print(
                    "===> Epoch[{}]: G_GAN:{:.4f}, image_loss:{:.4f}, LossG:{:.4f}, LossD:{:.4f}, penalty:{:.4f}, d_real:{:.4f}, d_fake:{:.4f}"
                    .format(epoch, loss_G_GAN.cpu(), (contentloss + image_loss).cpu(), loss_G.cpu(), loss_D.cpu(),
                            gradient_penalty.cpu(), d_loss_real.cpu(), d_loss_fake.cpu()))

                f = open(FilePath, 'a')
                f.write(
                    "===> Epoch[{}]: G_GAN:{:.4f}, image_loss:{:.4f}, LossG:{:.4f}, LossD:{:.4f}, d_real_loss:{:.6f}, d_fake_loss:{:.6f}, penalty:{:.4f}"
                    .format(epoch, loss_G_GAN.cpu(), image_loss.cpu(), loss_G.cpu(), loss_D.cpu(), d_loss_real.cpu(),
                            d_loss_fake.cpu(),
                            gradient_penalty.cpu()) + '\n')
                f.close()
                fake_out_save = transforms.ToPILImage()(deblur_fake.cpu()[0])
                fake_out_save.save('./pictureShow/fake_out.png')
                blur_save = transforms.ToPILImage()(deblur_real.cpu()[0])
                blur_save.save('./pictureShow/deblur_save.png')
                deblur_save = transforms.ToPILImage()(blur_real.cpu()[0])
                deblur_save.save('./pictureShow/blur_save.png')

        if epoch % 10 == 0:
            torch.save(netG.state_dict(), './result/netG_epoch_%d.pth' % (epoch))
            torch.save(netD.state_dict(), './result/netD_epoch_%d.pth' % (epoch))


criterion = torch.nn.L1Loss(size_average=True)
criterion = criterion.to(0)

if __name__ == '__main__':
    data_loader = CreateDataLoader(args)
    train(data_loader)



