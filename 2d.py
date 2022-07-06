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

images = {}
for i in range(30):
    img = labeled_images[i].get('image')
    id_ = labeled_images[i].get('id')
    images[id_] = ((img - img.min()) / (img.max() - img.min())).astype('float')
for i in range(20):
    img = unlabeled_images[i].get('image')
    id_ = unlabeled_images[i].get('id')
    images[id_] = ((img - img.min()) / (img.max() - img.min())).astype('float')
print("\nData loaded successfully. Total patients:", len(images))


## verify normalize
#print('Images:')
#for p_id in images.keys():
#    print(str(p_id) + ":", images.get(p_id).min(), "-", images.get(p_id).max())

# //////////////////////////////////// Args /////////////////////////////////////////////

class Args:
    def __init__(self):
        self.lr = 0.001
        self.epochs = 20
        self.bs = 16
        self.loss = 'mse'
        self.load_model = False
        self.initial_epoch = 0
        self.int_steps = 7
        self.int_downsize = 2
        self.run_name = '2d_bs16'
        self.model_dir = './trained-models/torch/' + self.run_name + '/'

args = Args()
os.makedirs(args.model_dir, exist_ok=False)


# //////////////////////////////////// DataLoader /////////////////////////////////////////////

class OneDirDataset(Dataset):
    def __init__(self, images, dis):
        self.data = []
        for p_id, p_imgs in images.items():
            for i in range(p_imgs.shape[0] - dis):
                self.data.append((p_imgs[i], p_imgs[i + dis]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        outputs = [torch.tensor(d).unsqueeze(-1) for d in self.data[index]]

        return tuple(outputs)

dataset = OneDirDataset(images, dis=1)
dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, pin_memory=False,num_workers=0)

# ///////////////////////////////////// loss ////////////////////////////////////////////
if args.loss == 'ncc':
    sim_loss_func = vxm.losses.NCC().loss
elif args.loss == 'mse':
    sim_loss_func = vxm.losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)


# /////////////////////////////////////// model //////////////////////////////////////////

enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]

if args.load_model:
    model = vxm.networks.VxmDense.load(args.load_model, device)
else:
    model = vxm.networks.VxmDense(
        inshape=(512, 512),
        nb_unet_features=[enc_nf, dec_nf],
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    )

model.to(device)
_ = model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


# ///////////////////////////////////// train ////////////////////////////////////////////

loss_history = []

for epoch in range(args.initial_epoch, args.epochs):

    # save model checkpoint
    if (epoch + 1) % 50 == 0:
        model.save(os.path.join(args.model_dir, '%04d.pt' % epoch))

    epoch_loss = 0
    epoch_length = 0
    epoch_start_time = time.time()

    for inputs in dataloader:
        # shape = (bs, 1, W, H)
        [moving_img, fixed_img] = [d.to(device).float().permute(0, 3, 1, 2) for d in inputs]

        # predict
        moved_img, flow = model(moving_img, fixed_img, registration=True)

        # calculate loss
        loss = sim_loss_func(fixed_img, moved_img)

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss * args.bs
        epoch_length += args.bs

    epoch_loss /= epoch_length

    # print epoch info
    msg = 'epoch %d/%d, ' % (epoch + 1, args.epochs)
    msg += 'loss= %.4e, ' % (epoch_loss)
    msg += 'time= %.4f ' % (time.time() - epoch_start_time)
    print(msg, flush=True)

    loss_history.append(epoch_loss.detach().cpu())

# final model save
model.save(os.path.join(args.model_dir, '%04d.pt' % args.epochs))

plt.plot(loss_history)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(args.model_dir + args.run_name + '.png')
plt.show()

# ///////////////////////////////////// evaluate ////////////////////////////////////////////

print("\nEvaluation started.")
patients_loss = []
evaluation_start_time = time.time()
k = 4

for p_id, p_imgs in images.items():
    p_loss = 0
    p_slices = 0
    a = torch.tensor(p_imgs).unsqueeze(1).to(device).float()

    for i in range((p_imgs.shape[0] - 1) // k):
        # shape = (bs, 1, W, H)
        moving_img = a[i * k: (i + 1) * k]
        fixed_img = a[i * k + 1: (i + 1) * k + 1]

        # predict
        moved_img, flow = model(moving_img, fixed_img, registration=True)

        # calculate loss
        loss = sim_loss_func(fixed_img, moved_img)

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        p_loss += loss * k
        p_slices += k

    patients_loss.append((p_loss / p_slices).detach().cpu())

# print evaluation info
msg = 'loss= %.4e, ' % (sum(patients_loss) / len(patients_loss))
msg += 'time= %.4f ' % (time.time() - evaluation_start_time)
print(msg, flush=True)
