import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorRegresser(nn.Module):
    def __init__(self, num_primary_color):
        super(ColorRegresser, self).__init__()
        in_dim = 3 # input: target_img ( bn, 3ch, h, w )
        out_dim = num_primary_color * 3  # primary_color_layersと同じものが出力になる bn, ln, 3, h, w
        self.num_primary_color = num_primary_color

        self.conv1 = nn.Conv2d(in_dim, in_dim * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_dim * 4, in_dim * 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_dim * 16, in_dim * 64, kernel_size=3, stride=2, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(in_dim * 4)
        self.bn2 = nn.BatchNorm2d(in_dim * 16)
        self.bn3 = nn.BatchNorm2d(in_dim * 64)

        self.fc1 = nn.Linear(in_dim * 64, in_dim * 32)
        self.fc2 = nn.Linear(in_dim * 32, in_dim * 16)
        self.fc3 = nn.Linear(in_dim * 16, num_primary_color * 64 * 64)

    def forward(self, target_img):
        resized = F.adaptive_avg_pool2d(target_img, (64, 64))
        h = self.bn1(F.relu(self.conv1(resized)))
        h = self.bn2(F.relu(self.conv2(h)))
        h = self.bn3(F.relu(self.conv3(h)))
        h = F.adaptive_avg_pool2d(h, (1,1)).view(h.size(0), -1) # bn, in_dim * 8
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        out = self.fc3(h).view(-1, self.num_primary_color, 64*64) #ln x HW
        out = F.softmax(out, dim=2)


        resized = resized.view(-1, 3, 64*64).permute(0,2,1) # bn, hw, 3

        out = torch.bmm(out, resized) # bn, ln, 3

        out = out.view(-1, self.num_primary_color, 3, 1, 1) * torch.ones_like(target_img).unsqueeze(1)

        # 0~1の範囲でクリップした方が良いかもしれない
        out = torch.clamp(out, min=0., max=1.)


        return out



class MaskGenerator(nn.Module):

    def __init__(self, num_primary_color):
        super(MaskGenerator, self).__init__()
        in_dim = 3 + num_primary_color * 3 # ex. 21 ch (= 3 + 6 * 3)
        out_dim = num_primary_color # num_out_layers is the same as num_primary_color.

        self.conv1 = nn.Conv2d(in_dim, in_dim * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_dim * 2, in_dim * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_dim * 4, in_dim * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.deconv1 = nn.ConvTranspose2d(in_dim * 8, in_dim * 4, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_dim * 8, in_dim * 2, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_dim * 4, in_dim * 2, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1)
        self.conv4 = nn.Conv2d(in_dim * 2 + 3, in_dim, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(in_dim * 2)
        self.bn2 = nn.BatchNorm2d(in_dim * 4)
        self.bn3 = nn.BatchNorm2d(in_dim * 8)
        self.bnde1 = nn.BatchNorm2d(in_dim * 4)
        self.bnde2 = nn.BatchNorm2d(in_dim * 2)
        self.bnde3 = nn.BatchNorm2d(in_dim * 2)
        self.bn4 = nn.BatchNorm2d(in_dim)

    def forward(self, target_img, primary_color_pack):
        x = torch.cat((target_img, primary_color_pack), dim=1)

        h1 = self.bn1(F.relu(self.conv1(x))) # *2
        h2 = self.bn2(F.relu(self.conv2(h1))) # *4
        h3 = self.bn3(F.relu(self.conv3(h2))) # *8
        h4 = self.bnde1(F.relu(self.deconv1(h3))) # *4
        h4 = torch.cat((h4, h2), 1) # *8
        h5 = self.bnde2(F.relu(self.deconv2(h4))) # *2
        h5 = torch.cat((h5, h1), 1) # *4
        h6 = self.bnde3(F.relu(self.deconv3(h5))) # *2
        h6 = torch.cat((h6, target_img), 1) # *2+3
        h7 = self.bn4(F.relu(self.conv4(h6)))

        return torch.sigmoid(self.conv5(h7)) # box constraint for alpha layers



class ResiduePredictor(nn.Module):
    def __init__(self, num_primary_color):
        super(ResiduePredictor, self).__init__()

        in_dim = 3 + num_primary_color * 4 # target画像を入力に追加している．alpha_layersをch方向に結合している
        out_dim = num_primary_color * 3

        self.conv1 = nn.Conv2d(in_dim, in_dim * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_dim * 2, in_dim * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_dim * 4, in_dim * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.deconv1 = nn.ConvTranspose2d(in_dim * 8, in_dim * 4, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_dim * 8, in_dim * 2, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_dim * 4, in_dim * 2, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1)
        self.conv4 = nn.Conv2d(in_dim * 2 + 3, in_dim, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(in_dim * 2)
        self.bn2 = nn.BatchNorm2d(in_dim * 4)
        self.bn3 = nn.BatchNorm2d(in_dim * 8)
        self.bnde1 = nn.BatchNorm2d(in_dim * 4)
        self.bnde2 = nn.BatchNorm2d(in_dim * 2)
        self.bnde3 = nn.BatchNorm2d(in_dim * 2)
        self.bn4 = nn.BatchNorm2d(in_dim)

    def forward(self, target_img, mono_color_layers_pack):
        # caution: mono_color_layers_pack is different from primary_color_pack.
        x = torch.cat((target_img, mono_color_layers_pack), dim=1)

        h1 = self.bn1(F.relu(self.conv1(x))) # *2
        h2 = self.bn2(F.relu(self.conv2(h1))) # *4
        h3 = self.bn3(F.relu(self.conv3(h2))) # *8
        h4 = self.bnde1(F.relu(self.deconv1(h3))) # *4
        h4 = torch.cat((h4, h2), 1) # *8
        h5 = self.bnde2(F.relu(self.deconv2(h4))) # *2
        h5 = torch.cat((h5, h1), 1) # *4
        h6 = self.bnde3(F.relu(self.deconv3(h5))) # *2
        h6 = torch.cat((h6, target_img), 1) # *2+3
        h7 = self.bn4(F.relu(self.conv4(h6)))

        residue_pack = torch.tanh(self.conv5(h7)) # box constraint for alpha layers
        # 各レイヤーの各RGBチャンネルごとに，residueの平均が0になるように正規化する
        residue_pack = residue_pack - residue_pack.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)

        return residue_pack # residue(pack) only
