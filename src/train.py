from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from net import MaskGenerator, ResiduePredictor
from mydataset import MyDataset
import cv2
import os

parser = argparse.ArgumentParser(description='baseline')
parser.add_argument('--run_name', type=str, default='debug', help='run-name. This name is used for output folder.')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--num_primary_color', type=int, default=6,
                    help='num of layers')
parser.add_argument('--rec_loss_lambda', type=float, default=1.0,
                    help='reconst_loss lambda')
parser.add_argument('--m_loss_lambda', type=float, default=1.0,
                    help='m_loss_lambda')
parser.add_argument('--sparse_loss_lambda', type=float, default=1.0,
                    help='sparse_loss lambda')
parser.add_argument('--distance_loss_lambda', type=float, default=1.0,
                    help='distance_loss_lambda')

parser.add_argument('--save_layer_train', type=int, default=1,
                    help='save_layer_train')


parser.add_argument('--num_workers', type=int, default=8,
                    help='num_workers of dataloader')
parser.add_argument('--csv_path', type=str, default='places.csv', help='path to csv of images path')

parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--reconst_loss_type', type=str, default='l1', help='[mse | l1 | vgg]')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# 出力先のフォルダーを作成
try:
    os.makedirs('results/%s' % args.run_name)
except OSError:
    pass




torch.manual_seed(args.seed)
cudnn.benchmark = True

device = torch.device("cuda" if args.cuda else "cpu")

train_dataset = MyDataset(args.csv_path, args.num_primary_color, mode='train')
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    worker_init_fn=lambda x: np.random.seed(),
    drop_last=True,
    pin_memory=True
    )


val_dataset = MyDataset(args.csv_path, args.num_primary_color, mode='val')
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    )




mask_generator = MaskGenerator(args.num_primary_color).to(device)
residue_predictor = ResiduePredictor(args.num_primary_color).to(device)


params = list(mask_generator.parameters())
params += list(residue_predictor.parameters())


optimizer = optim.Adam(params, lr=1e-3, betas=(0.0, 0.99))


# loss
def reconst_loss(reconst_img, target_img, type='mse'):
    if type == 'mse':
        loss = F.mse_loss(reconst_img, target_img.detach())
    elif type == 'l1':
        loss = F.l1_loss(reconst_img, target_img.detach())
    elif loss == 'vgg':
        pass

    return loss

def sparse_loss(alpha_layers):
    # alpha_layers: bn, ln, 1, h, w
    #print('alpha_layers.mean().item(): ', alpha_layers.mean().item())
    alpha_layers = alpha_layers.sum(dim=1, keepdim=True) / (alpha_layers * alpha_layers).sum(dim=1, keepdim=True)
    loss = F.l1_loss(alpha_layers, torch.ones_like(alpha_layers).to(device))
    return loss

def temp_distance(primary_color_layers, alpha_layers, rgb_layers):
    """
    　共分散行列をeye(3)とみなして簡単にしたもの．
    　３次元空間でのprimary_colorへのユークリッド距離を
    　ピクセルごとに算出し，alpha_layersで重み付けし，
    　最後にsum

    　primary_color_layers, rgb_layers: (bn, ln, 3, h, w)
    """
    diff = (primary_color_layers - rgb_layers)
    distance = (diff * diff).sum(dim=2, keepdim=True) # out: (bn, ln, 1, h, w)
    #loss = (distance * alpha_layers).sum(dim=1, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True).view(-1)
    loss = (distance * alpha_layers).sum(dim=1, keepdim=True).mean()
    #print('temp loss: ', loss)
    return loss # shape = (bn)





def squared_mahalanobis_distance_loss(primary_color_layers, alpha_layers, rgb_layers):
    """
     実装していない
     No implement of squared_mahalanobis_distance_loss
    """
    loss = temp_distance(primary_color_layers, alpha_layers, rgb_layers)
    return loss

def alpha_normalize(alpha_layers):
    # constraint (sum = 1)
    # layersの状態で受け取り，その形で返す. bn, ln, 1, h, w
    return alpha_layers / (alpha_layers.sum(dim=1, keepdim=True) + 1e-8)

def read_backimage():
    img = cv2.imread('backimage.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2,0,1))
    img = img/255
    img = torch.from_numpy(img.astype(np.float32))

    return img.view(1,3,256,256).to(device)

backimage = read_backimage()


def mono_color_reconst_loss(mono_color_reconst_img, target_img):
    loss = F.l1_loss(mono_color_reconst_img, target_img.detach())

    return loss


def train(epoch):
    mask_generator.train()
    residue_predictor.train()


    train_loss = 0
    for batch_idx, (target_img, primary_color_layers) in enumerate(train_loader):
        target_img = target_img.to(device) # bn, 3ch, h, w
        primary_color_layers = primary_color_layers.to(device)
        #primary_color_layers = primary_color_layers.to(device) # bn, num_primary_color, 3ch, h, w

        optimizer.zero_grad()


        # networkにforwardにする
        primary_color_pack = primary_color_layers.view(target_img.size(0), -1 , target_img.size(2), target_img.size(3))
        pred_alpha_layers_pack = mask_generator(target_img, primary_color_pack)
        #print('pred_alpha_layers_pack.size():', pred_alpha_layers_pack.size())

        # MaskGの出力をレイヤーごとにviewする
        pred_alpha_layers = pred_alpha_layers_pack.view(target_img.size(0), -1, 1, target_img.size(2), target_img.size(3))

        # 正規化などのprocessingを行う
        processed_alpha_layers = alpha_normalize(pred_alpha_layers)

        # mono_color_layers_packの作成．ひとつのtensorにしておく．
        #mono_color_layers = primary_color_layers * processed_alpha_layers
        mono_color_layers = torch.cat((primary_color_layers, processed_alpha_layers), 2) #shape: bn, ln, 4, h, w
        mono_color_layers_pack = mono_color_layers.view(target_img.size(0), -1 , target_img.size(2), target_img.size(3))

        # ResiduePredictorの出力をレイヤーごとにviewする
        residue_pack  = residue_predictor(target_img, mono_color_layers_pack)
        residue = residue_pack.view(target_img.size(0), -1, 3, target_img.size(2), target_img.size(3))
        #pred_unmixed_rgb_layers = mono_color_layers + residue * processed_alpha_layers
        pred_unmixed_rgb_layers = torch.clamp((primary_color_layers + residue), min=0., max=1.0)# * processed_alpha_layers

        #pred_unmixed_rgb_layers_pack = residue_predictor(target_img, mono_color_layers_pack)
        #pred_unmixed_rgb_layers = pred_unmixed_rgb_layers_pack.view(target_img.size(0), -1, 3, target_img.size(2), target_img.size(3))

        # alpha addしてreconst_imgを作成する
        #reconst_img = (pred_unmixed_rgb_layers * processed_alpha_layers).sum(dim=1)
        reconst_img = (pred_unmixed_rgb_layers * processed_alpha_layers).sum(dim=1)
        mono_color_reconst_img = (primary_color_layers * processed_alpha_layers).sum(dim=1)

        #print('reconst_img.size(): ', reconst_img.size())
        #print('pred_unmixed_rgb_layers.size(): ', pred_unmixed_rgb_layers.size())
        #print('primary_color_layers.size(): ', primary_color_layers.size())
        #print('processed_alpha_layers.size(): ', processed_alpha_layers.size())
        # Culculate loss.
        r_loss = reconst_loss(reconst_img, target_img, type=args.reconst_loss_type) * args.rec_loss_lambda
        m_loss = mono_color_reconst_loss(mono_color_reconst_img, target_img) * args.m_loss_lambda
        s_loss = sparse_loss(processed_alpha_layers) * args.sparse_loss_lambda
        #print('total_loss: ', total_loss)
        d_loss = squared_mahalanobis_distance_loss(primary_color_layers.detach(), processed_alpha_layers, pred_unmixed_rgb_layers) * args.distance_loss_lambda

        total_loss = r_loss + m_loss + s_loss + d_loss
        total_loss.backward()
        train_loss += total_loss.item()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('')
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(target_img), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                total_loss.item() / len(target_img)))
            print('reconst_loss *lambda: ', r_loss.item() / len(target_img))
            print('sparse_loss *lambda: ', s_loss.item() / len(target_img))
            print('squared_mahalanobis_distance_loss *lambda: ', d_loss.item() / len(target_img))


            for save_layer_number in range(args.save_layer_train):
                save_image(primary_color_layers[save_layer_number,:,:,:,:],
                       'results/%s/train_ep_' % args.run_name + str(epoch) + '_ln_%02d_primary_color_layers.png' % save_layer_number)
                #save_image(mono_color_layers[save_layer_number,:,:,:,:] + backimage * (1 - processed_alpha_layers[save_layer_number,:,:,:,:]),
                #       'results/%s/ep_' % args.run_name + str(epoch) + '_ln_%02d_mono_color_layers.png' % save_layer_number)
                save_image(primary_color_layers[save_layer_number,:,:,:,:] * processed_alpha_layers[save_layer_number,:,:,:,:] + backimage * (1 - processed_alpha_layers[save_layer_number,:,:,:,:]),
                       'results/%s/train_ep_' % args.run_name + str(epoch) + '_ln_%02d_mono_color_layers.png' % save_layer_number)
                save_image(pred_unmixed_rgb_layers[save_layer_number,:,:,:,:] * processed_alpha_layers[save_layer_number,:,:,:,:] + backimage * (1 - processed_alpha_layers[save_layer_number,:,:,:,:]),
                       'results/%s/train_ep_' % args.run_name + str(epoch) + '_ln_%02d_pred_unmixed_rgb_layers.png' % save_layer_number)
                save_image(reconst_img[save_layer_number,:,:,:].unsqueeze(0),
                       'results/%s/train_ep_' % args.run_name + str(epoch) + '_ln_%02d_reconst_img.png' % save_layer_number)
                save_image(mono_color_reconst_img[save_layer_number,:,:,:].unsqueeze(0),
                       'results/%s/train_ep_' % args.run_name + str(epoch) + '_ln_%02d_mono_color_reconst_img.png' % save_layer_number)
                save_image(target_img[save_layer_number,:,:,:].unsqueeze(0),
                       'results/%s/train_ep_' % args.run_name + str(epoch) + '_ln_%02d_target_img.png' % save_layer_number)


    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    # save model
    torch.save(mask_generator.state_dict(), 'results/%s/mask_generator.pth' % args.run_name)
    torch.save(residue_predictor.state_dict(), 'results/%s/residue_predictor.pth' % args.run_name)



def val(epoch):
    mask_generator.eval()
    residue_predictor.eval()


    with torch.no_grad():
        for batch_idx, (target_img, primary_color_layers) in enumerate(val_loader):
            target_img = target_img.to(device) # bn, 3ch, h, w
            primary_color_layers = primary_color_layers.to(device)

            primary_color_pack = primary_color_layers.view(target_img.size(0), -1 , target_img.size(2), target_img.size(3))
            pred_alpha_layers_pack = mask_generator(target_img, primary_color_pack)
            pred_alpha_layers = pred_alpha_layers_pack.view(target_img.size(0), -1, 1, target_img.size(2), target_img.size(3))
            processed_alpha_layers = alpha_normalize(pred_alpha_layers)
            mono_color_layers = torch.cat((primary_color_layers, processed_alpha_layers), 2) #shape: bn, ln, 4, h, w
            mono_color_layers_pack = mono_color_layers.view(target_img.size(0), -1 , target_img.size(2), target_img.size(3))
            residue_pack  = residue_predictor(target_img, mono_color_layers_pack)
            residue = residue_pack.view(target_img.size(0), -1, 3, target_img.size(2), target_img.size(3))
            pred_unmixed_rgb_layers = torch.clamp((primary_color_layers + residue), min=0., max=1.0)
            reconst_img = (pred_unmixed_rgb_layers * processed_alpha_layers).sum(dim=1)
            mono_color_reconst_img = (primary_color_layers * processed_alpha_layers).sum(dim=1)

            # batchsizeは１で計算されているはず．それぞれ保存する．
            save_layer_number = 0
            save_image(primary_color_layers[save_layer_number,:,:,:,:],
                   'results/%s/val_ep_' % args.run_name + str(epoch) + '_idx_%02d_primary_color_layers.png' % batch_idx)
            save_image(primary_color_layers[save_layer_number,:,:,:,:] * processed_alpha_layers[save_layer_number,:,:,:,:] + backimage * (1 - processed_alpha_layers[save_layer_number,:,:,:,:]),
                   'results/%s/val_ep_' % args.run_name + str(epoch) + '_idx_%02d_mono_color_layers.png' % batch_idx)
            save_image(pred_unmixed_rgb_layers[save_layer_number,:,:,:,:] * processed_alpha_layers[save_layer_number,:,:,:,:] + backimage * (1 - processed_alpha_layers[save_layer_number,:,:,:,:]),
                   'results/%s/val_ep_' % args.run_name + str(epoch) + '_idx_%02d_pred_unmixed_rgb_layers.png' % batch_idx)
            save_image(reconst_img[save_layer_number,:,:,:].unsqueeze(0),
                   'results/%s/val_ep_' % args.run_name + str(epoch) + '_idx_%02d_reconst_img.png' % batch_idx)
            save_image(mono_color_reconst_img[save_layer_number,:,:,:].unsqueeze(0),
                   'results/%s/val_ep_' % args.run_name + str(epoch) + '_idx_%02d_mono_color_reconst_img.png' % batch_idx)
            save_image(target_img[save_layer_number,:,:,:].unsqueeze(0),
                   'results/%s/val_ep_' % args.run_name + str(epoch) + '_idx_%02d_target_img.png' % batch_idx)

            if batch_idx == 1:
                break # trainloaderを使っているとき用にセット



if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        print('Start training')
        train(epoch)
        val(epoch)
