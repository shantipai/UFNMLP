import torch
import torch.nn as nn
import torch.nn.functional as F

def H(x, psfH):
    X = torch.fft.fft2(x, dim = (-2, -1))
    res = psfH * X
    return torch.fft.ifft2(res, dim = (-2, -1)).real

def Ht(x, psfHconj):
    X = torch.fft.fft2(x, dim = (-2, -1))
    res = psfHconj*X
    return torch.fft.ifft2(res, dim = (-2, -1)).real

def normalize_image(image):
    out_shape = image.shape
    image_flat = image.reshape((out_shape[0],out_shape[1]*out_shape[2]*out_shape[3]))
    image_max,_ = torch.max(image_flat,1)
    image_max_eye = torch.eye(out_shape[0], dtype = torch.float32, device=image.device)*1/image_max
    image_normalized = torch.reshape(torch.matmul(image_max_eye, image_flat), (out_shape[0],out_shape[1],out_shape[2],out_shape[3]))
    
    return image_normalized
class double_conv(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.d_conv(x)
        return x


class Unet(torch.nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()

        self.dconv_down1 = double_conv(in_ch, 6)
        self.dconv_down2 = double_conv(6, 12)
        self.dconv_down3 = double_conv(12, 24)
        self.dconv_down4 = double_conv(24, 48)
        self.dconv_down5 = double_conv(48, 96)
        self.dconv_down6 = double_conv(96, 192)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(24, 12, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(12, 6, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.upsample3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(48, 24, kernel_size = 2, stride = 2),
            torch.nn.ReLU(inplace=True)
        )
        self.upsample4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(96, 48, kernel_size = 2, stride = 2),
            torch.nn.ReLU(inplace = True)
        )
        self.upsample5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(192, 96, kernel_size = 2, stride = 2),
            torch.nn.ReLU(inplace = True)
        )
        self.dconv_up5 = double_conv(96 + 96, 96)
        self.dconv_up4 = double_conv(48 + 48, 48)
        self.dconv_up3 = double_conv(24 + 24, 24)
        self.dconv_up2 = double_conv(12 + 12, 12)
        self.dconv_up1 = double_conv(6 + 6, 6)

        self.conv_last = nn.Conv2d(6, out_ch, 1)
        self.afn_last = nn.Tanh()

    def forward(self, x):
        b, c, h_inp, w_inp = x.shape
        hb, wb = 64, 64
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        inputs = x
        #conv1 = 6
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        #conv2 = 12
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        #conv3 = 24
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        conv5 = self.dconv_down5(x)
        x = self.maxpool(conv5)

        conv6 = self.dconv_down6(x)

        x = self.upsample5(conv6)
        x = torch.cat([x, conv5], dim=1)

        x = self.dconv_up5(x)
        x = self.upsample4(x)
        x = torch.cat([x, conv4], dim=1)

        x = self.dconv_up4(x)
        x = self.upsample3(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample2(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample1(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)
        x = self.afn_last(x)
        out = x + inputs

        return out[:, :, :h_inp, :w_inp]


class Unet2(torch.nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Unet2, self).__init__()

        self.dconv_down1 = double_conv(in_ch, 6)
        self.dconv_down2 = double_conv(6, 12)
        self.dconv_down3 = double_conv(12, 24)
        self.dconv_down4 = double_conv(24, 48)
        self.dconv_down5 = double_conv(48, 96)
        self.dconv_down6 = double_conv(96, 192)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(24, 12, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(12, 6, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.upsample3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(48, 24, kernel_size = 2, stride = 2),
            torch.nn.ReLU(inplace=True)
        )
        self.upsample4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(96, 48, kernel_size = 2, stride = 2),
            torch.nn.ReLU(inplace = True)
        )
        self.upsample5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(192, 96, kernel_size = 2, stride = 2),
            torch.nn.ReLU(inplace = True)
        )
        self.dconv_up5 = double_conv(96 + 96, 96)
        self.dconv_up4 = double_conv(48 + 48, 48)
        self.dconv_up3 = double_conv(24 + 24, 24)
        self.dconv_up2 = double_conv(12 + 12, 12)
        self.dconv_up1 = double_conv(6 + 6, 6)

        self.conv_last = nn.Conv2d(6, out_ch, 1)
        self.conv_last2 = nn.Conv2d(in_ch, out_ch, 1)
        self.afn_last = nn.Tanh()

    def forward(self, x):
        b, c, h_inp, w_inp = x.shape
        hb, wb = 64, 64
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        inputs = x
        #conv1 = 6
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        #conv2 = 12
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        #conv3 = 24
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        conv5 = self.dconv_down5(x)
        x = self.maxpool(conv5)

        conv6 = self.dconv_down6(x)

        x = self.upsample5(conv6)
        x = torch.cat([x, conv5], dim=1)

        x = self.dconv_up5(x)
        x = self.upsample4(x)
        x = torch.cat([x, conv4], dim=1)

        x = self.dconv_up4(x)
        x = self.upsample3(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample2(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample1(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)
        x = self.afn_last(x)
        # change
        inputs =  self.conv_last2(inputs)
        out = x + inputs

        return out[:, :, :h_inp, :w_inp]


# aim x alpha1 alpha3




class Prior(torch.nn.Module):
    def __init__(self):
        super.__init__(Prior, self)
        self.body = Unet(3, 3)
        self.zero_cuda = torch.tensor(0, dtype = torch.float32, device=model.cuda_device)
    def forward(self, x, psfH, CTC, CTy, alpha1, alpha3, mu1, mu3):
        u = self.body(x)
        v = (alpha1 + mu1*H(x,psfH) + Cty)/(CTC + mu1)
        w = max(alpha3/mu3+x, self.zero_cuda)
        return u, v, w
        

class ADMM_Net(torch.nn.Module):
    def __init__(self, stage, padding, psf):
        super(ADMM_Net, self).__init__()
        _, psfh, psfw = psf.shape
        self.paddingx = psfh // 2
        self.paddingy = psfw //2
        self.CTCconv = torch.nn.Conv2d(15, 15, (3,3), 1, padding = 1)
        self.CTyconv = torch.nn.Conv2d(15, 15, (3,3), 1, padding = 1)
        self.alpha1conv = torch.nn.Conv2d(15, 15, (3, 3), 1, padding=1)
        self.alpha3conv = torch.nn.Conv2d(15, 15, (3, 3), 1, padding=1)
        self.mu1 = torch.nn.Parameter(torch.ones(stage, dtype = torch.float32))
        self.mu2 = torch.nn.Parameter(torch.ones(stage, dtype = torch.float32))
        self.mu3 = torch.nn.Parameter(torch.ones(stage, dtype = torch.float32))
        self.denosier = torch.nn.ModuleList([])
        self.h_var = torch.nn.Parameter(F.pad(psf, padding, 'constant', 0), 
                                            requires_grad=False)
        self.zero_cuda = torch.tensor(0, dtype = torch.float32)
        
        self.hshift = torch.fft.ifftshift(self.h_var, (-2, -1))
        
        
        #print('self.h_complex shape {}'.format(self.h_complex.shape))
        self.H = torch.fft.fft2(self.hshift, dim = (-2, -1))   
        self.Hconj =  self.H.conj()
        #print('self.Hconj shape {}'.format(self.Hconj.shape))
        self.HtH = torch.abs(self.H * self.Hconj)
        self.xconv = torch.nn.Conv2d(15, 15, (3, 3), 1, padding=1)
        self.stage = stage
        for i in range(stage):
            self.denosier.append(Unet(15, 15))
        self.fusion = torch.nn.Conv2d(15, 3, (3, 3), 1, 1)
        self.res = torch.nn.Sequential(torch.nn.Conv2d(3, 3, (3, 3,), 1, 1),
                                       torch.nn.ReLU(),
                                       torch.nn.Conv2d(3, 3, (3, 3), 1, 1))
        self.end = Unet(15, 15)

    def Initialize(self, y):
        y = y.repeat(1, 5, 1, 1)
        CTC = torch.ones_like(y, dtype = torch.float32).to(y.device)
        CTy = y.to(y.device)
        #CTC = self.CTCconv(CTC)
        #CTy = self.CTyconv(CTy)
        alpha1 = self.alpha1conv(torch.zeros_like(CTy, dtype = torch.float32))
        alpha3 = self.alpha3conv(torch.zeros_like(CTy, dtype = torch.float32))
        x = self.xconv(CTy)
        return CTC, CTy, alpha1, alpha3, x
    
    def forward(self, y):
        b, c, h, w = y.shape
        CTC, CTy, alpha1, alpha3, x = self.Initialize(y)
        
        hl = self.paddingx
        hr = (self.paddingx+768)
        wl = self.paddingy
        wr = (self.paddingy+768)
        for i in range(self.stage):
            u = self.denosier[i](x)
            vk = (CTC + self.mu1[i])
            v = (alpha1 + self.mu1[i]*H(x,self.H) + CTy)/vk
            w = torch.maximum(alpha3/self.mu3[i]+x, self.zero_cuda)
            r = (self.mu3[i] * w - alpha3) + self.mu2[i] * u + Ht(self.mu1[i]*v - alpha1 , self.Hconj)
            rft = torch.fft.fft2(r, dim = (-2, -1))
            xft = rft/(self.mu1[i] * self.HtH + self.mu2[i] + self.mu3[i])
            x = torch.fft.ifft2(xft, dim = (-2, -1)).real
            alpha1 = alpha1 + self.mu1[i]*(H(x,self.H) - v)
            alpha3 = alpha3 +self.mu3[i] * (x-w)

            normalize_image(x)

        x = x[:,:,hl:hr, wl: wr]
        
        
        x = self.end(x)
        x = self.fusion(x)
        x = self.res(x)
        #print(x.shape)
        return x

        

