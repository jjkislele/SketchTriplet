import os
from utils import *

modelf = 'flickr15k_1904041458'
feat_photo_path = 'out_feat/{}/feat.npz'.format(modelf)
feat_sketch_path = 'out_feat/{}/feat_sketch.npz'.format(modelf)
feat_photo = np.load(feat_photo_path)
feat_sketch = np.load(feat_sketch_path)
retrievalf = 'out_feat/{}/result'.format(modelf)

os.makedirs(retrievalf, exist_ok=True)

feat_s = feat_sketch['feat_sketch']
cls_name_s = feat_sketch['cls_name']
cls_num_s = feat_sketch['cls_num']
path_s = feat_sketch['path']
name_s = feat_sketch['name']

feat_p = feat_photo['feat']
cls_name_p = feat_photo['cls_name']
cls_num_p = feat_photo['cls_num']
path_p = feat_photo['path']
name_p = feat_photo['name']

AP = []

for i in range(0, len(feat_s)):
    user = path_s[i].split('/')[-2]
    sketch = name_s[i]
    userp = os.path.join(retrievalf, user)
    os.makedirs(userp, exist_ok=True)

    # compute similarity by L2
    # d_{12} = \sqrt {\sum_{k = 1}^{n} (x_{1k} - x_{2k})^2}
    dist_l2 = np.sqrt(np.sum(np.square(feat_s[i] - feat_p), 1))
    order = np.argsort(dist_l2)

    # write in retrieval results
    order_cls_name_p = cls_name_p[order]
    order_name_p = name_p[order]
    order_cls_num_p = cls_num_p[order]
    order_score = dist_l2[order]
    fid = open('{}/{}'.format(userp, name_s[i]), 'a')
    for j in range(0, len(feat_p)):
        fid.write(f'{order_score[j]} {order_cls_name_p[j]}/{order_name_p[j]}\n')
    fid.close()

    ap = compute_AP(sketch, order_cls_num_p)
    AP.append(ap)
    print(f"[{i}/{len(feat_s)}] completed. AP: {ap}")

print(f"Done! mAP: {ap.mean()}")