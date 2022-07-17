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
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage.transform import resize
from collections import Counter

os.environ['VXM_BACKEND'] = 'pytorch'
os.environ['NEURITE_BACKEND'] = 'pytorch'
import voxelmorph as vxm
import neurite as ne

device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.deterministic = True


# ////////////////////////////////////////// load & normalize ///////////////////////////////////////

labeled_images = np.load('/home/adeleh/MICCAI-2022/UMIS-data/medical-data/synaps/labeled_images.npy', allow_pickle=True)
unlabeled_images = np.load('/home/adeleh/MICCAI-2022/UMIS-data/medical-data/synaps/unlabeled_images.npy', allow_pickle=True)

train_images = []
train_labels = []
test_images = []
test_labels = []
for i in range(0, 30):
    if i == 3 or i == 8:
        continue
    lb = labeled_images[i].get('label')
    for j in range(lb.shape[0]):
        if 6 in lb[j, :, :]:
            lb = lb[j + 5:j + 30, :, :]
            lb = np.where(lb == 6, np.ones_like(lb), np.zeros_like(lb))
            lb = resize(lb, (25, 256, 256), anti_aliasing=False)
            lb = ((lb - lb.min()) / (lb.max() - lb.min())).astype('float')
            img = labeled_images[i].get('image')[j + 5:j + 30, :, :]
            img = resize(img, (25, 256, 256), anti_aliasing=True)
            img = ((img - img.min()) / (img.max() - img.min())).astype('float')
            if i < 20:
                train_images.append(img)
                train_labels.append(lb)
            else:
                test_images.append(img)
                test_labels.append(lb)
            break
print("\nData loaded successfully.")


# //////////////////////////////////// Args /////////////////////////////////////////////

class Args:
    def __init__(self):
        self.lr = 0.001
        self.epochs = 50
        self.bs = 16
        self.loss = 'mse'
        self.seg_w = 0.1
        self.smooth_w = 0.01
        self.load_model = False
        self.initial_epoch = 0
        self.int_steps = 7
        self.int_downsize = 2
        self.run_name = '2d_supervised'
        self.model_dir = '/home/adeleh/MICCAI-2022/armin/master-thesis/trained-models/256x256/' + self.run_name + '/'

args = Args()
os.makedirs(args.model_dir, exist_ok=True)


# //////////////////////////////////// DataLoader /////////////////////////////////////////////

class OneDirDataset(Dataset):
    def __init__(self, images, labels, dis):
        self.data = []
        for p_imgs, p_lbs in zip(images, labels):
            for i in range(p_imgs.shape[0] - dis):
                if p_lbs is None:
                    self.data.append(((p_imgs[i], p_imgs[i + dis]), None))
                else:
                    self.data.append(((p_imgs[i], p_imgs[i + dis]), (p_lbs[i], p_lbs[i + dis])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_output = tuple([torch.tensor(d).unsqueeze(-1) for d in self.data[index][0]])
        lb_output = tuple([torch.tensor(d).unsqueeze(-1) for d in self.data[index][1]])

        return img_output, lb_output

train_dataset = OneDirDataset(train_images, train_labels, dis=1)
train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, pin_memory=False,num_workers=0)

test_dataset = OneDirDataset(test_images, test_labels, dis=1)
# test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=True, pin_memory=False,num_workers=0)

# ///////////////////////////////////// loss ////////////////////////////////////////////
if args.loss == 'ncc':
    sim_loss_func = vxm.losses.NCC().loss
elif args.loss == 'mse':
    sim_loss_func = vxm.losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

smooth_loss_func = vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss
seg_loss_func = vxm.losses.Dice().loss

# /////////////////////////////////////// model //////////////////////////////////////////

enc_nf = [16, 32, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]

if args.load_model:
    model = vxm.networks.VxmDense.load(args.load_model, device)
else:
    model = vxm.networks.VxmDense(
        inshape=(256, 256),
        nb_unet_features=[enc_nf, dec_nf],
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    )

model.to(device)
_ = model.train()

print('number of all params:', sum(p.numel() for p in model.parameters()))
print('number of trainable params:', sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


# ///////////////////////////////////// train ////////////////////////////////////////////

loss_history = []
sim_loss_history = []
seg_loss_history = []
smooth_loss_history = []

for epoch in range(args.initial_epoch, args.epochs):

    # save model checkpoint
    if (epoch + 1) % 50 == 0:
        model.save(os.path.join(args.model_dir, '%04d.pt' % epoch))

    epoch_sim_loss = 0
    epoch_seg_loss = 0
    epoch_smooth_loss = 0
    epoch_loss = 0
    epoch_length = 0
    epoch_start_time = time.time()

    for inputs in train_dataloader:
        # shape = (bs, 1, W, H)
        imgs , lbs = inputs
        [moving_img, fixed_img] = [d.to(device).float().permute(0, 3, 1, 2) for d in imgs]
        [moving_lb, fixed_lb] = [d.to(device).float().permute(0, 3, 1, 2) for d in lbs]

        # predict
        moved_img, flow = model(moving_img, fixed_img, registration=True)
        moved_lb = model.transformer(moving_lb, flow)

        # calculate loss
        sim_loss = sim_loss_func(fixed_img, moved_img)
        seg_loss = seg_loss_func(fixed_lb, moved_lb)
        smooth_loss = smooth_loss_func(_, flow)
        loss = sim_loss + args.seg_w * seg_loss + args.smooth_w * smooth_loss

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_sim_loss += sim_loss * args.bs
        epoch_seg_loss += seg_loss * args.bs
        epoch_smooth_loss += smooth_loss * args.bs
        epoch_loss += loss * args.bs
        epoch_length += args.bs

    epoch_sim_loss /= epoch_length
    epoch_seg_loss /= epoch_length
    epoch_smooth_loss /= epoch_length
    epoch_loss /= epoch_length

    # print epoch info
    msg = 'epoch %d/%d, ' % (epoch + 1, args.epochs)
    msg += 'loss= %.4e, ' % (epoch_loss)
    msg += 'sim_loss= %.4e, ' % (epoch_sim_loss)
    msg += 'seg_loss= %.4f, ' % (epoch_seg_loss)
    msg += 'smooth_loss= %.4e, ' % (epoch_smooth_loss)
    msg += 'time= %.4f ' % (time.time() - epoch_start_time)
    print(msg, flush=True)

    loss_history.append(epoch_loss.detach().cpu())
    sim_loss_history.append(epoch_sim_loss.detach().cpu())
    seg_loss_history.append(epoch_seg_loss.detach().cpu())
    smooth_loss_history.append(epoch_smooth_loss.detach().cpu())

# final model save
model.save(os.path.join(args.model_dir, '%04d.pt' % args.epochs))

plt.plot(range(len(loss_history)), loss_history, "-b", label="loss")
plt.plot(range(len(loss_history)), sim_loss_history, "-r", label="sim-loss")
plt.plot(range(len(loss_history)), seg_loss_history, "-g", label="seg-loss")
plt.plot(range(len(loss_history)), smooth_loss_history, "-c", label="smooth-loss")
plt.legend(loc="upper left")
plt.ylim(-1.5, 2.0)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(args.model_dir + args.run_name + '.png')
plt.show()

# ///////////////////////////////////// evaluate ////////////////////////////////////////////

def evaluate(images, labels):
    patients_sim_loss = []
    patients_seg_loss = []
    patients_smooth_loss = []
    patients_loss = []
    evaluation_start_time = time.time()
    k = 1

    for p_imgs, p_lbs in zip(images, labels):
        p_sim_loss = 0
        p_seg_loss = 0
        p_smooth_loss = 0
        p_loss = 0
        p_slices = 0
        imgs = torch.tensor(p_imgs).unsqueeze(1).to(device).float()
        lbs = torch.tensor(p_lbs).unsqueeze(1).to(device).float()

        for i in range((p_imgs.shape[0] - 1) // k):
            # shape = (bs, 1, W, H)
            moving_img = imgs[i * k: (i + 1) * k]
            fixed_img = imgs[i * k + 1: (i + 1) * k + 1]
            moving_lb = lbs[i * k: (i + 1) * k]
            fixed_lb = lbs[i * k + 1: (i + 1) * k + 1]

            # predict
            moved_img, flow = model(moving_img, fixed_img, registration=True)
            moved_lb = model.transformer(moving_lb, flow)

            # calculate loss
            sim_loss = sim_loss_func(fixed_img, moved_img)
            seg_loss = seg_loss_func(fixed_lb, moved_lb)
            smooth_loss = smooth_loss_func(_, flow)
            loss = sim_loss + args.seg_w * seg_loss + args.smooth_w * smooth_loss

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            p_sim_loss += sim_loss * k
            p_seg_loss += seg_loss * k
            p_smooth_loss += smooth_loss * k
            p_loss += loss * k
            p_slices += k

        patients_sim_loss.append((p_sim_loss / p_slices).detach().cpu())
        patients_seg_loss.append((p_seg_loss / p_slices).detach().cpu())
        patients_smooth_loss.append((p_smooth_loss / p_slices).detach().cpu())
        patients_loss.append((p_loss / p_slices).detach().cpu())

    # print evaluation info
    msg = 'loss= %.4e, ' % (sum(patients_loss) / len(patients_loss))
    msg += 'sim_loss= %.4e, ' % (sum(patients_sim_loss) / len(patients_sim_loss))
    msg += 'seg_loss= %.4f, ' % (sum(patients_seg_loss) / len(patients_seg_loss))
    msg += 'smooth_loss= %.4e, ' % (sum(patients_smooth_loss) / len(patients_smooth_loss))
    msg += 'time= %.4f ' % (time.time() - evaluation_start_time)
    print(msg, flush=True)

print("\nEvaluation for training-set started.")
evaluate(train_images, train_labels)

print("\nEvaluation for test-set started.")
evaluate(test_images, test_labels)
