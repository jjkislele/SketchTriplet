import os

import torch.nn as nn
from model.SketchTriplet import SketchTriplet
from model.SketchTriplet_half_sharing import SketchTriplet as SketchTriplet_hs
from model.SketchTriplet_half_sharing import BranchNet
from utils import *
from flickr15k_dataset import init_flickr15k_dataloader

# ---------------------------------
batchSize = 128                 # input batch size
niter = 500                     # number of epochs to train for
outf = 'out/flickr15k_'         # folder to output model checkpoints
outf = outf + time.strftime('%y%m%d%H%M', time.localtime(time.time()))
checkpoint = 50                 # checkpointing after batches
img_size = 256                  # resize the input image to square
root = '../deep_hashing'        # root path for dataset
# ---------------------------------

def train(epoch, dataloader, net, optimizer):
    accum_loss = 0
    net.train()
    for i, (img_a, img_p, img_n) in enumerate(dataloader):
        anc_src, pos_src, neg_src = Variable(img_a.cuda()), Variable(img_p.cuda()), Variable(img_n.cuda())

        net.zero_grad()
        feat_a, feat_p, feat_n = net(anc_src, pos_src, neg_src)
        loss = criterion(feat_a, feat_p, feat_n)

        loss.backward()
        optimizer.step()
        accum_loss += loss.data[0]

        print(f'[{epoch}][{i}/{len(dataloader)}] loss: {loss.data[0]:.4f}')
    return accum_loss / len(dataloader)

# setup dataloader ---------------------------------
os.makedirs('out', exist_ok=True)
os.makedirs(outf, exist_ok=True)
feed_random_seed()
train_loader, test_loader = init_flickr15k_dataloader(batchSize, img_size, root)
# setup net ---------------------------------
branch_net = BranchNet()
model = SketchTriplet_hs(branch_net)
model = model.cuda()
criterion = nn.TripletMarginLoss(margin=1.0, p=2.0)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=0.0005, momentum=0.9)

resume_epoch = 0

fid = open('{}/loss_and_accurary.txt'.format(outf), 'a')
for epoch in range(resume_epoch+1, niter+1):
    train_loss = train(epoch, train_loader, model, optimizer)
    print('[%d/%d] train loss: %f' % (epoch, niter, train_loss))
    fid.write('[%d/%d] train loss: %f\n' % (epoch, niter, train_loss))

    if epoch % checkpoint == 0:
        # no eval ???
        # save checkpoints
        torch.save(model.state_dict(), os.path.join(outf, f'{epoch:03d}.pth'))

fid.close()