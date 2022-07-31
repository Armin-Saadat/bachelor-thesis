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
organs = {0:"background", 1:"spleen", 2:"left_kidney", 3:"right_kidney", 6:"liver", 8:"aorta", 11:"pancreas"}
SELECTED_ORGAN = 11
print("\nselected organ:", organs[SELECTED_ORGAN])

labeled_images = np.load('/home/adeleh/MICCAI-2022/UMIS-data/medical-data/synaps/labeled_images.npy', allow_pickle=True)
unlabeled_images = np.load('/home/adeleh/MICCAI-2022/UMIS-data/medical-data/synaps/unlabeled_images.npy', allow_pickle=True)
unlabeled_images_starts = [55, 60, 100, 40, 40, 80, 80, 75, 95, 55, 100, 50, 45, 45, 110, 60, 65, 55, 45, 95]

train_images = []
train_labels = []
test_images = []
test_labels = []
for i in range(0, 30):
    if i == 3 or i == 8:
        continue
    lb = labeled_images[i].get('label')
    for j in range(lb.shape[0]):
        if SELECTED_ORGAN in lb[j, :, :]:
            lb = lb[j + 5:j + 30, :, :]
            lb = np.where(lb == SELECTED_ORGAN, np.ones_like(lb), np.zeros_like(lb))
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
for i in range(0, 20):
<<<<<<< HEAD
    break
    s = unlabeled_images_starts[i]
    img = unlabeled_images[i].get('image')[s:s + 25, :, :]
    img = resize(img, (25, 256, 256), anti_aliasing=True)
    img = ((img - img.min()) / (img.max() - img.min())).astype('float')
    train_images.append(img)
    train_labels.append(np.zeros_like(img))
=======
    imgs = labeled_images[i].get('image')
    for j in range((imgs.shape[0] - 30) // 25):
        img = imgs[j*25: j*25 + 25, :, :]
        img = resize(img, (25, 256, 256), anti_aliasing=True)
        img = ((img - img.min()) / (img.max() - img.min())).astype('float')
        train_images.append(img)
        train_labels.append(np.zeros_like(img))
for i in range(0, 20):
    imgs = unlabeled_images[i].get('image')
    for j in range((imgs.shape[0] - 30) // 25):
        img = imgs[j*25: j*25 + 25, :, :]
        img = resize(img, (25, 256, 256), anti_aliasing=True)
        img = ((img - img.min()) / (img.max() - img.min())).astype('float')
        train_images.append(img)
        train_labels.append(np.zeros_like(img))
>>>>>>> 49f67b6f93a41c2a6323eafa10597581deaf80a0
print("\nData loaded successfully.")
print("number of training subjects:", len(train_images))
print("number of testing subjects:", len(test_images))


# //////////////////////////////////// Args /////////////////////////////////////////////

class Args:
    def __init__(self):
        self.lr = 0.001
<<<<<<< HEAD
        self.epochs = 70
=======
        self.epochs = 50
>>>>>>> 49f67b6f93a41c2a6323eafa10597581deaf80a0
        self.bs = 24
        self.loss = 'mse'
        self.seg_w = 0.0
        self.smooth_w = 0.0001
        self.load_model = '/home/adeleh/MICCAI-2022/armin/master-thesis/trained-models/unet/organs/pancreas/0050.pt'
        self.initial_epoch = 50
        self.int_steps = 7
        self.int_downsize = 2
        self.run_name = 'test'
        self.model_dir = '/home/adeleh/MICCAI-2022/armin/master-thesis/trained-models/unet/' + self.run_name + '/'

args = Args()
os.makedirs(args.model_dir, exist_ok=True)

# assert args.bs == 24, "batch-size must be equal to number of pairs per patient."


# //////////////////////////////////// DataLoader /////////////////////////////////////////////

class OneDirDataset(Dataset):
    def __init__(self, images, labels, dis):
        self.data = []
        unlabeled_data = []
        for p_imgs, p_lbs in zip(images, labels):
            for i in range(p_imgs.shape[0] - dis):
                if p_lbs[i].max() != 0:
                    self.data.append(((p_imgs[i], p_imgs[i + dis]), (p_lbs[i], p_lbs[i + dis])))
                else:
                    unlabeled_data.append(((p_imgs[i], p_imgs[i + dis]), (p_lbs[i], p_lbs[i + dis])))
        for d in unlabeled_data:
            self.data.append(d)
        del unlabeled_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_output = tuple([torch.tensor(d).unsqueeze(-1) for d in self.data[index][0]])
        lb_output = tuple([torch.tensor(d).unsqueeze(-1) for d in self.data[index][1]])

        return img_output, lb_output

train_dataset = OneDirDataset(train_images, train_labels, dis=1)
train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False, pin_memory=False,num_workers=0)

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

f = open(args.model_dir + 'result.txt', "a")

for epoch in range(args.initial_epoch, args.epochs):

    # save model checkpoint
    if (epoch + 1) % 50 == 0:
        model.save(os.path.join(args.model_dir, '%04d.pt' % epoch))

    epoch_sim_loss = 0
    epoch_seg_loss = 0
    epoch_smooth_loss = 0
    epoch_loss = 0
    epoch_length = 0
    epoch_seg_count = 0
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

        if moving_lb.max() == 0:
            loss = sim_loss + args.smooth_w * smooth_loss
        else:
            loss = sim_loss + args.seg_w * seg_loss + args.smooth_w * smooth_loss
            epoch_seg_loss += seg_loss * args.bs
            epoch_seg_count += args.bs

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_sim_loss += sim_loss * args.bs
        epoch_smooth_loss += smooth_loss * args.bs
        epoch_loss += loss * args.bs
        epoch_length += args.bs

    epoch_sim_loss /= epoch_length
    if epoch_seg_count != 0:
        epoch_seg_loss /= epoch_seg_count
    else:
        epoch_seg_loss = torch.tensor(0)
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
    f.write(msg + '\n')    

    loss_history.append(epoch_loss.detach().cpu())
    sim_loss_history.append(epoch_sim_loss.detach().cpu())
    seg_loss_history.append(epoch_seg_loss.detach().cpu())
    smooth_loss_history.append(epoch_smooth_loss.detach().cpu())

# final model save
model.save(os.path.join(args.model_dir, '%04d.pt' % args.epochs))

figure, axis = plt.subplots(1, 4, figsize=(60, 15))
axis[0].plot(loss_history)
axis[0].set_title("Final Loss")
axis[1].plot(sim_loss_history)
axis[1].set_title("Similarity Loss")
axis[2].plot(seg_loss_history)
axis[2].set_title("Segmentation Loss")
axis[3].plot(smooth_loss_history)
axis[3].set_title("Smooth Loss")
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

    with torch.no_grad():
        for p_imgs, p_lbs in zip(images, labels):
            p_sim_loss = 0
            p_seg_loss = 0
            p_smooth_loss = 0
            p_loss = 0
            p_slices = 0
            p_slices_seg = 0
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
                if moving_lb.max() == 0:
                    loss = sim_loss + args.smooth_w * smooth_loss
                else:
                    loss = sim_loss + args.seg_w * seg_loss + args.smooth_w * smooth_loss
                    p_seg_loss += seg_loss * k
                    p_slices_seg += k

                p_sim_loss += sim_loss * k
                p_smooth_loss += smooth_loss * k
                p_loss += loss * k
                p_slices += k

            patients_sim_loss.append((p_sim_loss / p_slices).detach().cpu())
            if p_slices_seg != 0:
                patients_seg_loss.append((p_seg_loss / p_slices_seg).detach().cpu())
            patients_smooth_loss.append((p_smooth_loss / p_slices).detach().cpu())
            patients_loss.append((p_loss / p_slices).detach().cpu())

    if len(patients_seg_loss) == 0:
        patients_seg_loss.append(0)
    # print evaluation info
    msg = 'loss= %.4e, ' % (sum(patients_loss) / len(patients_loss))
    msg += 'sim_loss= %.4e, ' % (sum(patients_sim_loss) / len(patients_sim_loss))
    msg += 'seg_loss= %.4f, ' % (sum(patients_seg_loss) / len(patients_seg_loss))
    msg += 'smooth_loss= %.4e, ' % (sum(patients_smooth_loss) / len(patients_smooth_loss))
    msg += 'time= %.4f ' % (time.time() - evaluation_start_time)
    print(msg, flush=True)
    f.write('\nEvaluation Info:\n')    
    f.write(msg + '\n')    

print("\nEvaluation for training-set started.")
evaluate(train_images, train_labels)

print("\nEvaluation for test-set started.")
evaluate(test_images, test_labels)

f.close()
