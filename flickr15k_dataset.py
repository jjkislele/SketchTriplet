import os
import os.path
import torchvision.transforms as transforms
from random import choice
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class flickr15k_dataset(Dataset):
    def __init__(self, train=True, transforms=None):
        self.root = '../deep_hashing/dataset/Flickr_15K_edge2'
        self.gt_path = '../deep_hashing/groundtruth'
        self.gt = {}
        self.transforms = transforms

        for i in range(1, 34):
            self.gt[str(i)] = []
        file = open(self.gt_path)
        for line in file:
            sketch_cls = line.split()[0]
            img_path = line.split()[1][:-4] + '.jpg'
            img_cls = img_path.split('/')[0]
            img_name = img_path.split('/')[1][:-4]
            img_path = os.path.join(self.root, img_path)
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
        anc_fn = self.datapath[idx]
        cls_name_a = anc_fn[0]
        cls_num_a = anc_fn[1]
        anc_path = anc_fn[2]
        anc_name = anc_fn[3]
        img_a = Image.open(anc_path).convert('RGB')

        # select pos
        pos_fn = self.gt[cls_num_a][choice([i for i in range(0, len(self.gt[cls_num_a])) if i not in [int(anc_name)]])]
        cls_name_p = pos_fn[1]
        cls_num_p = cls_num_a
        pos_path = pos_fn[0]
        pos_name = pos_fn[2]
        img_p = Image.open(pos_path).convert('RGB')
        # select neg
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

if __name__ == '__main__':
    test_set = flickr15k_dataset(train=False)
    test_set[1]
    print(len(test_set))