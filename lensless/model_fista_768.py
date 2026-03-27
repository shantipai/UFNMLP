import torch
from torch import nn, fft
from math import sqrt, radians, tan
import torch.nn.functional as F
from torch.nn import init

#from .unet import UNet
from .spt import SPFST
from .admm_net_5psf import Unet as Unet_ad
import numpy as np


# Initialize network weights
def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)



class Spatial_AM(nn.Module):
    def __init__(self, alpha):
        super(Spatial_AM, self).__init__()
        self.net = nn.Conv2d(2, 1, 3, 1, padding=1)
        #self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.alpha = alpha
    def forward(self, x):
        # 0,1
        x = x.transpose(0, 1)  # 9,3,1280,1280
        ave_out = torch.mean(x, dim=1, keepdim=True)   # 9,1,1280,1280
        max_out = torch.max(x, dim=1, keepdim=True)[0]  # 9,1,1280,1280
        out = torch.cat([ave_out, max_out], dim=1)  # 9,2,1280,1280
        out = self.sigmoid(self.net(out))  # 9,1,1280,1280
        return (x*(1-self.alpha) + x * out*self.alpha).transpose(0, 1)

class FISTA_unfold(nn.Module):
    def __init__(self, LayerNo, alpha): # threshold=lamda  # add L
        super(FISTA_unfold, self).__init__()
        self.padding_psf = (400, 400, 400, 400)
        self.padding_img = (260, 260, 260, 260)

        self.denoise = torch.nn.ModuleList([])
        self.LayerNo = LayerNo

        self.denoise.append(Spatial_AM(alpha))
        self.denoise.append(Unet_ad(9, 9))

    def forward(self, x0, y1, b, mu, threshold, t, i, phi):
        # FISTA
        phi = ft(phi)
        phiT = phi.conj()
        PhiTPhi = torch.abs(phi * phiT)  # 3,5,h,w
        F_b = ft(b)
        PhiTb = phiT* F_b

        # gradient descent
        y1_cut = F.pad(y1, self.padding_img, mode='constant', value=0)
        grad = ift(PhiTPhi* ft(y1_cut)).real - ift(PhiTb).real
        # new size
        grad = grad[..., 260:1060, 260:1060]
        z1 = y1 - mu * grad

        # denoise
        z1 = self.denoise[0](z1)
        z1 = self.denoise[1](z1)

        # proximal
        x1 = torch.sign(z1) * torch.maximum(torch.abs(z1) - mu * threshold, torch.tensor(0.0))

        # update
        y2 = x1 + t*(x1-x0)

        return x1, y2

class WeightedReconstruct(nn.Module):
    def __init__(self):
        super(WeightedReconstruct, self).__init__()
        self.center = 660  #644
        self.f = 2e-3
        self.pixel_size = 2e-6
        #self.directions = [[-15,-15],[-15,-10],[15,-10],[15,15],[0,0],[-20,0],[0,-20],[0,20],[20,0]]
        self.directions = [[0,0],[-20,-20],[-20,20],[20,-20],[20,20],[-20,0],[0,-20],[20,0],[0,20]]  # ,[-20,0],[0,-20],[20,0],[0,20]

    def cal_weight(self,i,j, sigma):
        device = sigma.device
        center_x = torch.tensor([self.center + self.f*tan(radians(i))/self.pixel_size], device=device)  # , device=device
        center_y = torch.tensor([self.center + self.f * tan(radians(j)) / self.pixel_size], device=device)
        # grid = torch.arange(1288).float()
        grid = torch.arange(1320).float()
        x, y = torch.meshgrid(grid, grid, indexing="ij")
        x = x.to(device)
        y = y.to(device)
        # sigma = 1000
        sigma = sigma * 100
        gaussian_kernel = torch.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.max()
        return gaussian_kernel

    def forward(self, x, sigma):
        # cal gauss weight
        weight_list = []
        weight_sum = 0
        count = 0
        for i,j in self.directions:
            weight = self.cal_weight(i,j, sigma[count])
            weight_sum += weight[..., 260:1060, 260:1060]
            weight_list.append(weight[..., 260:1060, 260:1060])
            count += 1
        # normalize
        normalized_weight = [weight/weight_sum for weight in weight_list]

        bc,width,h,w = x.size()
        device = x.device
        #out = torch.zeros([bc, 1, h, w], dtype = torch.float32, device=device)
        out = 0
        for i in range(width):
            #weight = self.weight[i].to(device)
            out = out + x[:, i, :, :] * normalized_weight[i]

        out = out.unsqueeze(1)

        return out


class FISTANet(nn.Module):
    def __init__(self, LayerNo, width, psf, device):
        super(FISTANet, self).__init__()
        self.LayerNo = LayerNo  # num of unfolding layers
        self.padding_psf = (400, 400, 400, 400)
        psf = F.pad(psf, self.padding_psf, mode='constant', value=0)
        self.psf =  torch.nn.Parameter(psf.to(device), requires_grad=True)  # 520
        _, _, h, w = psf.shape
        #print('device:')
        #print(self.psf.device)

        # weight params
        self.alpha = nn.Parameter(torch.tensor(0.05, device=device))
        self.sigma = nn.Parameter(torch.ones(width, dtype = torch.float32, device=device) * torch.tensor(10.0))
        self.threshold = nn.Parameter(torch.ones(self.LayerNo, dtype=torch.float32, device=device) * torch.tensor(0.01))  # 0.01
        self.mu = nn.Parameter(torch.ones(self.LayerNo, dtype=torch.float32, device=device) * torch.tensor(1e-3))  # 1e-3
        self.t = nn.Parameter(torch.ones(self.LayerNo, dtype=torch.float32, device=device) * torch.tensor(1))

        self.unfold = nn.ModuleList([FISTA_unfold(LayerNo=self.LayerNo, alpha=self.alpha) for _ in range(LayerNo)])  # phi,b均为

        self.decode_9to1 = WeightedReconstruct()
        self.width = width
        self.spt = SPFST(inDim=1, outDim=1, dim=16, numBlocks = [2, 2, 2])
        #self.unet2 = Unet_ad(1, 1)
        #self.decode_5to1 = torch.nn.Sequential(
            #nn.Conv2d(self.width, 16, 3, padding=1),
            #nn.LeakyReLU(),
        #    nn.Conv2d(width, 1, 1),
        #)

        initialize_weights(self)

        self.Sp = nn.Softplus()

    def forward(self, image):
        b, c, h, w = image.shape
        image = image.reshape(b * c, 1, h, w)  # 3,1,h,w
        image = image.repeat(1, self.width, 1, 1)
        x0 = image[..., 260:1060, 260:1060]
        bx, cx, hx, wx = x0.shape

        y1 = x0
        for i in range(self.LayerNo):
            x0, y1 = self.unfold[i](x0, y1, image, self.mu[i], self.threshold[i], self.t[i], i, self.psf)  # self.t[i]  # , loss

        #x = self.decode_5to1(x0)
        #x = x + self.unet(x)
        x = self.decode_9to1(x0, self.sigma)
        #x = self.unet2(x)
        x = self.spt(x)
        x = torch.sigmoid(x.reshape((b, c, hx, wx)))

        return x, self.sigma[0]


def ft(image):
    return fft.fft2(fft.ifftshift(image, dim=(-2, -1)), norm='ortho')
    #return fft.rfft2(fft.ifftshift(image, dim=(-2, -1)), norm='ortho')
def ift(image):
    return fft.fftshift(fft.ifft2(image, norm='ortho'), dim=(-2, -1))
    #return fft.fftshift(fft.irfft2(image, norm='ortho'), dim=(-2, -1))
