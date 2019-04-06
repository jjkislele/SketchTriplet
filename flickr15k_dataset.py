import os
import os.path
from random import choice
import random
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class flickr15k_dataset(Dataset):
    def __init__(self, train=True, root='../deep_hashing', transforms=None):
        self.root = root
        self.gt_path = os.path.join(self.root, 'groundtruth')
        self.img_set_path = os.path.join(self.root, 'dataset/Flickr_15K_edge2')
        self.sketch_set_path = os.path.join(self.root, 'dataset/330sketches')
        self.gt = {}
        self.transforms = transforms

        for i in range(1, 34):
            """flickr15k has 34 classes, dump way"""
            self.gt[str(i)] = []
        file = open(self.gt_path)
        for line in file:
            sketch_cls = line.split()[0]
            img_path = line.split()[1][:-4] + '.jpg'
            img_cls = img_path.split('/')[0]
            img_name = img_path.split('/')[1][:-4]
            img_path = os.path.join(self.img_set_path, img_path)
            # check img exist
            if os.path.exists(img_path):
                self.gt[sketch_cls].append((img_path, img_cls, img_name))
        file.close()

        self.datapath = []
        for i in range(1, 34):
            item = str(i)
            for fn in self.gt[item]:
                # item: class number
                # f[0]: file absolute path
                # f[1]: class name
                # f[2]: file name
                self.datapath.append((fn[1], item, fn[0], fn[2]))

    def __getitem__(self, idx):
        # select anchor -> sketch
        anc_fn = self.datapath[idx]
        cls_name_a = anc_fn[0]
        cls_num_a = anc_fn[1]
        anc_path = anc_fn[2]
        anc_name = anc_fn[3]
        sketch_path = random_select_sketch(cls_num_a, self.sketch_set_path)
        img_a = Image.open(sketch_path).convert('RGB')

        # select pos    -> photography edge(preprocessed by Canny edge detection)
        pos_fn = self.gt[cls_num_a][choice([i for i in range(0, len(self.gt[cls_num_a])) if i not in [int(anc_name)]])]
        cls_name_p = pos_fn[1]
        cls_num_p = cls_num_a
        pos_path = pos_fn[0]
        pos_name = pos_fn[2]
        img_p = Image.open(pos_path).convert('RGB')
        # select neg    -> photography edge
        cls_num_n = str(choice([i for i in range(1,len(self.gt)+1) if i not in [cls_num_a]]))
        neg_fn = self.gt[cls_num_n][choice([i for i in range(0, len(self.gt[cls_num_n]))])]
        cls_name_n = neg_fn[1]
        neg_name = neg_fn[2]
        neg_path = neg_fn[0]
        img_n = Image.open(neg_path).convert('RGB')

        img_a = self.transforms(img_a)
        img_p = self.transforms(img_p)
        img_n = self.transforms(img_n)

        return img_a, img_p, img_n

    def __len__(self):
        return len(self.datapath)

    def get_single_img_edge(self, idx):
        img_edge = self.datapath[idx]
        cls_name = img_edge[0]
        cls_num = img_edge[1]
        path = img_edge[2]
        name = img_edge[3]
        img_edge_src = Image.open(path).convert('RGB')
        img_edge_src = self.transforms(img_edge_src)
        return img_edge_src, img_edge


def random_select_sketch(cls_num, sketch_set_path):
    """select sketch randomly"""
    """flickr15K has 10 users"""
    user_select = str(random.randint(1,10))
    sketch_path = os.path.join(sketch_set_path, user_select, (cls_num+'.png'))
    return sketch_path

def init_flickr15k_dataloader(batchSize, img_size):
    """load dataset"""
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    train_loader = DataLoader(flickr15k_dataset(train=True, transforms=transform),
                              batch_size=batchSize, shuffle=True, num_workers=4, pin_memory=True)
    print(f'train set: {len(train_loader.dataset)}')
    test_loader = DataLoader(flickr15k_dataset(train=False, transforms=transform),
                             batch_size=batchSize, shuffle=False, num_workers=4, pin_memory=True)
    print(f'val set: {len(test_loader.dataset)}')

    return train_loader, test_loader

def init_flickr15k_dataset(img_size):
    """load dataset"""
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    return flickr15k_dataset(train=False, transforms=transform)

if __name__ == '__main__':
    test_set = flickr15k_dataset(train=False)
    test_set[1]
    print(len(test_set))