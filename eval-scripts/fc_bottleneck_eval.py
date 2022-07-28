# ////////////////////////////////////////// imports ///////////////////////////////////////
import os, sys
import glob
import time
import numpy as np
import shutil
import imageio
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage.transform import resize

os.environ['VXM_BACKEND'] = 'pytorch'
os.environ['NEURITE_BACKEND'] = 'pytorch'
import voxelmorph as vxm
import neurite as ne

device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

# ////////////////////////////////////////// load & normalize ///////////////////////////////////////

labeled_images = np.load('/home/adeleh/MICCAI-2022/UMIS-data/medical-data/synaps/labeled_images.npy', allow_pickle=True)
unlabeled_images = np.load('/home/adeleh/MICCAI-2022/UMIS-data/medical-data/synaps/unlabeled_images.npy', allow_pickle=True)

organs = {0:"background", 1:"spleen", 2:"left_kidney", 3:"right_kidney", 6:"liver", 8:"aorta", 11:"pancreas"}
SELECTED_ORGAN = 6
print("\nselected organ:", organs[SELECTED_ORGAN])

images = {}
labels = {}
for i in range(20, 30):
    lb = labeled_images[i].get('label')
    id_ = labeled_images[i].get('id')
    for j in range(lb.shape[0]):
        if SELECTED_ORGAN in lb[j, :, :]:
            lb = lb[j + 5:j + 30, :, :]
            lb = np.where(lb == SELECTED_ORGAN, np.ones_like(lb), np.zeros_like(lb))
            lb = resize(lb, (25, 256, 256), anti_aliasing=False)
            lb = ((lb - lb.min()) / (lb.max() - lb.min())).astype('float')
            img = labeled_images[i].get('image')[j + 5:j + 30, :, :]
            img = resize(img, (25, 256, 256), anti_aliasing=True)
            img = ((img - img.min()) / (img.max() - img.min())).astype('float')
            images[id_] = img
            labels[id_] = lb
            break
print("\nData loaded successfully. Total patients:", len(images))
number_of_patients = len(images)


# //////////////////////////////////// Args /////////////////////////////////////////////

class Args:
    def __init__(self):
        self.bs = 1
        self.loss = 'dice'
        self.load_model = "/home/adeleh/MICCAI-2022/armin/master-thesis/trained-models/fc_bottleneck/organs/liver/0070.pt"
        self.dis = 1
        self.int_steps = 7
        self.int_downsize = 2

args = Args()


# ///////////////////////////////////// loss ////////////////////////////////////////////

class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        y_true = torch.where(y_true > 0, torch.ones_like(y_true), torch.zeros_like(y_true))
        y_pred = torch.where(y_pred > 0, torch.ones_like(y_pred), torch.zeros_like(y_pred))
        intersect = torch.sum(y_pred * y_true)
        sum_ = torch.sum(y_true) + torch.sum(y_pred)
        if sum_ == 0:
            return False
        dice = (2 * intersect) / sum_
        return dice

if args.loss == 'ncc':
    sim_loss_func = vxm.losses.NCC().loss
elif args.loss == 'mse':
    sim_loss_func = vxm.losses.MSE().loss
elif args.loss == 'dice':
    sim_loss_func = Dice().loss
else:
    raise ValueError('loss should be "mse" or "ncc" or "dice", but found "%s"' % args.image_loss)


# /////////////////////////////////////// model //////////////////////////////////////////

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)

        return out


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class MyUnet(nn.Module):
    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):

        super().__init__()

        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            enc_nf = [16, 32, 32, 32]
            dec_nf = [32, 32, 32, 32, 32, 16, 16]
            nb_features = [enc_nf, dec_nf]

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x, task, x_history_=None):

        # encoder forward pass
        if task == 'encode':
            x_history = [x]
            for level, convs in enumerate(self.encoder):
                for conv in convs:
                    x = conv(x)
                x_history.append(x)
                x = self.pooling[level](x)

            return x, x_history

        # decoder forward pass with upsampling and concatenation
        elif task == 'decode':
            x_history = x_history_
            assert x_history is not None, "x_history_ is None."
            for level, convs in enumerate(self.decoder):
                for conv in convs:
                    x = conv(x)
                if not self.half_res or level < (self.nb_levels - 2):
                    x = self.upsampling[level](x)
                    x = torch.cat([x, x_history.pop()], dim=1)

            # remaining convs at full resolution
            for conv in self.remaining:
                x = conv(x)

            return x


class FC_Bottleneck(nn.Module):
    def __init__(self, image_size):
        super(FC_Bottleneck, self).__init__()
        self.image_size = image_size
        self.ndims = len(image_size)

        enc_nf = [16, 32, 32, 32, 32]
        dec_nf = [32, 32, 32, 32, 32, 16, 16, 2]
        self.unet = MyUnet(inshape=image_size, infeats=2, nb_features=[enc_nf, dec_nf])

        self.input_size = self.hidden_size = 32 * 8 * 8
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=False)

        Conv = getattr(nn, 'Conv%dd' % self.ndims)
        self.flow = Conv(self.unet.final_nf, self.ndims, kernel_size=3, padding=1)

        self.spatial_transformer = SpatialTransformer(size=image_size)

    def forward(self, images, labels=None):
        # shape of imgs/lbs: (T, bs, 1, 256, 256)
        T, bs = images.shape[0], images.shape[1]

        # shape of encoder_out: (T-1, bs, 32, 8, 8)
        X, X_history = [], []
        for src, trg in zip(images[:-args.dis], images[args.dis:]):
            x, x_history = self.unet(torch.cat([src, trg], dim=1), 'encode')
            X.append(x.unsqueeze(0))
            X_history.append(x_history)
        encoder_out = torch.cat(X, dim=0)

        # shape of lstm_out: (T-1, bs, 32, 8, 8)
        device = 'cuda' if images.is_cuda else 'cpu'
        h_0 = torch.randn(1, bs, self.hidden_size).to(device)
        c_0 = torch.randn(1, bs, self.hidden_size).to(device)
        lstm_out, (h_n, c_n) = self.lstm(encoder_out.view(T-1, bs, -1), (h_0, c_0))
        lstm_out = lstm_out.view(T-1, bs, 32, 8, 8)

        # shape of flow: (T-1, bs, 2, 256, 256)
        Y = [self.unet(lstm_out[i], 'decode', X_history[i]).unsqueeze(0) for i in range(T-1)]
        flow = torch.cat(Y, dim=0)

        # shape of moved_images = (T-1, bs, 1, 256, 256)
        moved_images = torch.cat(
            [self.spatial_transformer(src, flow).unsqueeze(0) for src, flow in zip(images[:-args.dis], flow)], dim=0)

        if labels is not None:
            moved_labels = torch.cat(
                [self.spatial_transformer(src, flow).unsqueeze(0) for src, flow in zip(labels[:-args.dis], flow)], dim=0)
            return [moved_images, moved_labels, flow]
        else:
            return [moved_images, flow]


model = FC_Bottleneck((256, 256))
if args.load_model:
    snapshot = torch.load(args.load_model, map_location='cpu')
    model.load_state_dict(snapshot['model_state_dict'])
    print("model loaded successfully.")

model.to(device)
_ = model.eval()

print('number of all params:', sum(p.numel() for p in model.parameters()))
print('number of trainable params:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# shape = (40, 1, 1, 256, 256)
imgs = [torch.tensor(p_imgs).unsqueeze(1).unsqueeze(1) for p_imgs in images.values()]
lbs = [torch.tensor(p_lbs).unsqueeze(1).unsqueeze(1) for p_lbs in labels.values()]


# ///////////////////////////////////// evaluate ////////////////////////////////////////////

print("\nEvaluation started.")
epoch_loss = 0
epoch_length = 0
epoch_start_time = time.time()

with torch.no_grad():
    for k in range(number_of_patients // args.bs):
        # shape of input = (T, bs, 1, W, H)
        input_img = torch.cat(imgs[k:k + args.bs], dim=1).to(device).float()
        input_lb = torch.cat(lbs[k:k + args.bs], dim=1).to(device).float()
        k += args.bs

        # predict
        moved_images, moved_labels, flow = model(input_img, input_lb)

        # calculate loss
        loss = sim_loss_func(input_lb[args.dis:, ...], moved_labels)

        if loss == False:
            continue
        epoch_loss += loss * args.bs
        epoch_length += args.bs

    epoch_loss /= epoch_length

# print epoch info
if args.loss == 'dice':
    msg = 'dice-score= %.4f, ' % epoch_loss
else:
    msg = 'mse= %.4e, ' % epoch_loss
msg += 'time= %.4f ' % (time.time() - epoch_start_time)
print(msg, flush=True)
