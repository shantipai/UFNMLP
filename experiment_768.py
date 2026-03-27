#command:
# train:  path/to/your/dataset --train
# test: path/to/your/dataset --eval

# import
import torch
import numpy as np
import random
import os
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import time
from piq import psnr
from lensless.ssim_torch import ssim
import torchvision
#from pytorch_lightning.loggers import TensorBoardLogger
#from lensless.AG_EN import compute_average_gradient, calculate_entropy
from torch.nn.functional import mse_loss
import cv2
import torch.nn.functional as F

# import models
from lensless.model_fista_768 import FISTANet
from lensless.diffusercam_768 import LenslessLearningCollection
# float16
from torch.cuda.amp import GradScaler, autocast
import lpips
# flops
from fvcore.nn import FlopCountAnalysis


# random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(5)

# choose GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device('cuda:0')

def analyze_model_complexity(model, device):  # params and flops
    model.eval()
    dummy_input = torch.randn(1, 3, 1320, 1320).to(device)
    # cal_params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_params / 1e6:.2f} M")
    # cal FLOPs
    flops = FlopCountAnalysis(model, dummy_input)
    print(f"Flops: {flops.total() / 1e9:.2f} GFLOPs")

    return flops, total_params

def measure_inference_performance(model, device, dataloader, num_test=20):  # inference time and memory use
    model.eval()
    times = []
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for i, (diffused, gt, *_) in enumerate(dataloader):
            if i >= num_test:
                break
            diffused = diffused.to(device)
            # cal time
            torch.cuda.synchronize()
            start = time.time()
            out, _ = model(diffused)
            torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)

        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time

        # memory
        max_mem = torch.cuda.max_memory_allocated(device) / 1024**2  # MB

        print(f"Inference time: {avg_time*1000:.2f} ms")
        print(f"Max memory usage: {max_mem:.2f} MB")

        return avg_time, max_mem


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, default='dataset', help="Path to Lensless Learning dataset")
    parser.add_argument('--eval', action="store_true", help="Evaluate trainable models")
    parser.add_argument('--train', action="store_true", help="Fit trainable models")
    parser.add_argument('--lr', default=2e-4, help="learning rate")
    parser.add_argument('--checkpoint', type=bool, default=False, help="checkpoint")
    parser.add_argument('--ckpt_path', type=str, default=None, help="checkpoint path")
    parser.add_argument('--epoch_start', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=300)  # default = 5
    # parser.add_argument('--logs', default='/media/max525/ubuntu1/wby/lenless/FISTA_simple_re/weights', type=Path, help="Logs directory")
    parser.add_argument('--results', default='/media/max525/ubuntu1/wby/lenless/FISTA_simple_re/results_768/fista', type=Path, help="Results Image directory")
    parser.add_argument('--save_path', default='/media/max525/ubuntu1/wby/lenless/FISTA_simple_re/saved', type=Path, help="save checkpoints")
    parser.add_argument('--val_path', default='/media/max525/ubuntu1/wby/lenless/FISTA_simple_re/saved_val', type=Path, help="save val images")
    parser.add_argument('--use_edge', default=True, type=bool, help="use edge loss")
    return parser.parse_args()

def cal_edge(img):
    if img.dim() == 4:
        img = img[0]
    img = img.permute(1, 2, 0).cpu().detach().numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(img, 50, 200)
    # to tensor
    edge = np.expand_dims(edge, axis=-1)
    edge = torch.from_numpy(edge).permute(2, 0, 1).float() / 255.0
    return edge.to(device)


def train_model(args):
    # dataloader
    #args.dataset_path = '/media/max525/ubuntu1/data/new_tju/LSDIR_768'
    collection = LenslessLearningCollection(args.dataset_path, isTrain=True)  # train & val dataset, psf, ROI(1400/1536)
    train_dataset = collection.train_dataset
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=4, persistent_workers=True)
    val_dataset = collection.val_dataset
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=4, persistent_workers=True)

    # setup model
    model = FISTANet(LayerNo=5, width=9, psf=collection.psf, device=device)
    model = model.to(device)
    # load checkpoint
    if args.checkpoint:
        pth_path = '/media/max525/ubuntu1/wby/lenless/FISTA_simple_re/saved_768/all_model_072.pth'
        ckpt = torch.load(pth_path)
        model.load_state_dict(ckpt)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # lr scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, eta_min=1e-5)  # 2e-8
    scaler = GradScaler()

    # train
    psnr_best = 0
    for epoch in range(args.epoch_start, args.max_epochs):
        model.train()
        epoch_loss = 0
        start = time.time()

        for i, (diffused, gt, name) in enumerate(train_dataloader):
            optimizer.zero_grad()
            with autocast():
                diffused = diffused.to(device)
                gt = gt[..., 260:1060, 260:1060].to(device)
                out, sigma = model(diffused)

                if args.use_edge is True:
                    edge_out = cal_edge(collection.ROI(gt))
                    padding = (16, 16, 16, 16)
                    edge_out = F.pad(edge_out, padding, mode='constant', value=0)
                    MSE_loss = mse_loss(out * (1 + 0.05 * edge_out), gt * (1 + 0.05 * edge_out))
                else:
                    MSE_loss = mse_loss(out, gt)

                #LPIPS_loss = lpips_loss(out, gt)
                loss = MSE_loss #+ LPIPS_loss * 0.1
                epoch_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            if i%50 == 0:
                print('%4d:%4d/%4d loss = %.10f time = %s' % ((epoch), i, len(train_dataset), epoch_loss/(i+1), time.time()-start), flush=True)

        # epoch end
        print('epoch = %4d, loss = %.10f, time = %4.2f s' % ((epoch), epoch_loss / len(train_dataset), time.time()-start), flush=True)
        print('sigma= %.2f' % sigma)
        print(optimizer.param_groups[0]['lr'])

        # update lr
        scheduler.step()

        # validate
        model.eval()
        with torch.no_grad():
            val_psnr, val_ssim, pred, gt = validate_model(val_dataset, val_dataloader, model, collection)
            print('PSNR = %f SSIM = %f' % (val_psnr, val_ssim))   # .cpu().numpy()

            image_show = torch.cat([pred, gt], dim=-1)
            #logger.experiment.add_images(tag="Preds/Val", img_tensor=image_show.unsqueeze(0), global_step=epoch)
            #writer.add_images(tag="Preds/Val", img_tensor=image_show.unsqueeze(0), global_step=epoch)
            val_path = os.path.join(args.val_path, 'epoch%03d.png' % (epoch))
            write_images(image_show, val_path)

            # save best models
            if val_psnr > psnr_best:
                psnr_best = val_psnr
                torch.save(model.state_dict(), os.path.join(args.save_path, 'all_model_%03d.pth' % (epoch)))
            elif val_psnr <= psnr_best and psnr_best-val_psnr < 0.1:
                torch.save(model.state_dict(), os.path.join(args.save_path, 'all_model_weak_%03d.pth' % (epoch)))


def validate_model(val_dataset, val_dataloader, model, collection):
    all_psnr = 0
    all_ssim = 0
    for i, (diffused, gt, name) in enumerate(val_dataloader):
        diffused = diffused.to(device)
        gt = gt[..., 260:1060, 260:1060].to(device)
        with torch.no_grad():
            out, _ = model(diffused)

        out = collection.ROI(out)  #.unsqueeze(0)
        gt = collection.ROI(gt)
        PSNR = psnr(out, gt).item()
        out = out.squeeze()
        gt = gt.squeeze()
        SSIM = ssim(out, gt).item()
        all_psnr += PSNR
        all_ssim += SSIM
    val_psnr = all_psnr / len(val_dataset)
    val_ssim = all_ssim / len(val_dataset)

    return val_psnr, val_ssim, out, gt  #img return last one



def test_model(args):
    #args.dataset_path = '/media/max525/ubuntu1/data/new_tju/LSDIR_768'
    collection = LenslessLearningCollection(args.dataset_path, isTrain=False)
    test_dataset = collection.test_dataset
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=4, persistent_workers=True)

    model = FISTANet(LayerNo=5, width=9, psf=collection.psf, device=device)
    model = model.to(device)

    # lpips
    lpips_loss = lpips.LPIPS(net='alex').to(device)
    lpips_loss.eval()

    # load checkpoint
    pth_path = args.ckpt_path
    ckpt = torch.load(pth_path)
    model.load_state_dict(ckpt, strict=False)
    '''
    print("===Model Complexity===")
    analyze_model_complexity(model, device)
    print("===Inference Performance===")
    measure_inference_performance(model, device, test_dataloader)
    '''

    # test
    output = []
    model.eval()
    with torch.no_grad():
        with autocast():
            for i, (diffused, gt, img_name) in enumerate(test_dataloader):
                diffused = diffused.to(device)  # .unsqueeze(0)
                gt = gt[..., 260:1060, 260:1060].to(device)  # .unsqueeze(0)
                out, _ = model(diffused)

                # cal index
                out = collection.ROI(out)
                gt = collection.ROI(gt)
                output.append({
                    'num': i,
                    'psnr': psnr(out, gt).item(),
                    'ssim': ssim(out.squeeze(), gt.squeeze()).item(),
                    'lpips': lpips_loss(out, gt).item(),
                })
                # save imgs
                save_path = os.path.join(args.results, img_name[0])
                write_images(out, save_path)

    # write index
    with open('test_index.txt', 'w') as f:
        avg_psnr = 0
        avg_ssim = 0
        avg_lpips = 0
        for dict_item in output:
            f.write(str(dict_item)+'\n')
            # cal avg
            avg_psnr = avg_psnr + dict_item['psnr']
            avg_ssim = avg_ssim + dict_item['ssim']
            avg_lpips = avg_lpips + dict_item['lpips']

        avg_psnr = avg_psnr / len(output)
        avg_ssim = avg_ssim / len(output)
        avg_lpips = avg_lpips / len(output)

        f.write('psnr=' + str(avg_psnr)+'\n''ssim=' + str(avg_ssim)+'\n''lpips=' + str(avg_lpips)+'\n')


def write_images(image, save_path):
    image = (image.cpu() * 255).byte()  # .to(torch.int16)
    image = torch.squeeze(image, 0)
    torchvision.io.write_png(image, save_path)


# main
#def main():
if __name__ == '__main__':
    print('Main')
    args = parse_arguments()

    if args.train:
        print('Training...')
        train_model(args)

    if args.eval:
        test_model(args)


    print("Done")