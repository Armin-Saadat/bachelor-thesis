# ////////////////////////////////////////// imports ///////////////////////////////////////
import os, sys
import glob
import time
import png
import numpy as np
import shutil
import imageio
import pickle
import random
import torch
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
print(len(images))


## verify normalize
#print('Images:')
#for p_id in images.keys():
#    print(str(p_id) + ":", images.get(p_id).min(), "-", images.get(p_id).max())


# //////////////////////////////////// Args /////////////////////////////////////////////

class Args():
    def __init__(self):
        self.lr = 0.001
        self.epochs = 10
        self.bs = 20
        self.loss = 'mse'
        self.load_model = False
        self.initial_epoch = 0
        self.int_steps = 7
        self.int_downsize = 2
        self.model_dir = './trained-models/torch/1/'

args = Args()
os.makedirs(args.model_dir, exist_ok=False)


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
    if epoch + 1 % 50 == 0:
        model.save(os.path.join(args.model_dir, '%04d.pt' % epoch))

    epoch_loss = 0    
    volume_count = 0
    epoch_start_time = time.time()        

    for p_id, p_imgs in images.items():
        volume_loss = 0        
        a = torch.tensor(p_imgs).unsqueeze(1).to(device).float()                
        print(a.shape)        
        
        volume_slices = 0
        for i in range((p_imgs.shape[0] - 1) // args.bs):            
            #shape = (bs, 1, W, H)
            moving_img = a[i*args.bs: (i+1) * args.bs]
            fixed_img = a[i*args.bs + 1 : (i+1) * args.bs + 1]
        
            # predict
            moved_img, flow = model(moving_img, fixed_img, registration=True)        

            # calculate loss                
            loss = sim_loss_func(fixed_img, moved_img)

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            volume_loss += loss * args.bs
            volume_slices += args.bs
        
        epoch_loss += volume_loss / volume_slices 
        volume_count += 1      

    # print epoch info  
    msg = 'epoch %d/%d, ' % (epoch + 1, args.epochs)
    msg += 'loss= %.4e, ' % (epoch_loss / volume_count)
    msg += 'time= %.4f, ' % (time.time() - epoch_start_time)
    print(msg, flush=True)

    loss_history.append(epoch_loss / volume_count)

# final model save
model.save(os.path.join(args.model_dir, '%04d.pt' % args.epochs))

plt.plot(loss_history)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig("loss_history.png")
plt.show()




