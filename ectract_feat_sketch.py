import os

from model.SketchTriplet_half_sharing import BranchNet
from model.SketchTriplet_half_sharing import SketchTriplet as SketchTriplet_hs

from utils import *
from flickr15k_dataset import init_flickr15k_dataset

# ---------------------------------
modelf = 'flickr15k_1904041458'                 # folder to output model
model_path = 'out/{}/500.pth'.format(modelf)    # absolute path to model
outf = 'out_feat/{}'.format(modelf)             # folder to output features
img_size = 256                                  # resize the input image to square
# ---------------------------------


# setup dataset ---------------------------------
os.makedirs('out_feat', exist_ok=True)
os.makedirs(outf, exist_ok=True)
flickr15k_dataset = init_flickr15k_dataset(img_size)

# setup net     ---------------------------------
branch_net = BranchNet()                        # for photography edge
model = SketchTriplet_hs(branch_net)
model.load_state_dict(torch.load(model_path))
model = model.cuda()
model.eval()

out = np.empty((len(flickr15k_dataset.sketch_datapath),branch_net.num_feat), dtype=np.float32)
                                                # len(sketch_datapath): 330
                                                # num_feat: 100
out_cls_name = []
out_cls_num = []
out_path = []
out_name = []

for i in range(len(flickr15k_dataset.sketch_datapath)):
    sketch_src, sketch_info = flickr15k_dataset.get_single_sketch(i)
    sketch = Variable(sketch_src.unsqueeze(0)).cuda()
    feat = model.get_branch_sketch(sketch)
    feat = feat.cpu().data.numpy()
    out[i] = feat
    cls_name = sketch_info[0]
    cls_num = sketch_info[1]
    path = sketch_info[2]
    name = sketch_info[3]
    out_cls_name.append(cls_name)
    out_cls_num.append(cls_num)
    out_path.append(path)
    out_name.append(name)
    print(f"[{i}/{len(flickr15k_dataset.sketch_datapath)}] ('{cls_name}', '{cls_num}', '{name}') completed")

np.savez(os.path.join(outf, 'feat_sketch.npz'), feat_sketch=out, cls_name=cls_name, cls_num=cls_num, path=path, nane=name)
print("Done!")