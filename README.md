# Triplet_Loss_SBIR in Pytorch
This repo contains code implemented by Pytorch for the T.Bui et al's paper "[Compact Descriptors for Sketch-based Image Retrieval using a Triplet loss Convolutional Neural Network](https://doi.org/10.1016/j.cviu.2017.06.007)"[[Repo](https://github.com/TuBui/Triplet_Loss_SBIR)|[Page](http://www.cvssp.org/data/Flickr25K/CVIU16.html)].

The difference of the perposed network's architecture confuses me. In the paper, shown as Figure. 1, in each branch, the ``conv4`` layer don't have ``ReLu`` node right behind it, though, in [original codes](https://github.com/TuBui/Triplet_Loss_SBIR/blob/master/models/train.prototxt) ``conv4`` does.
I consult the [original codes](https://github.com/TuBui/Triplet_Loss_SBIR/blob/master/models/train.prototxt) to build the net.

The network seems able to reproduce the results though, there is still much room for improvement IN MY CODE:

- The modified triplet loss function proposed by the paper doesn't have implementations yet. Default triplet loss function from ``torch.nn.TripletMarginLoss`` is used indeed.
- The mentioned training policy is not implemented.
- Eval process is not included. All photographs from ***flickr15k*** are used for training. I know it's a bad idea.

## Dependency
- pytorch 0.4.0 with torchvision 0.2.1
- python 3.6.4 
- anaconda 4.4.10 recommend

## How to Run
First, the pretrained model based on Flickr15k can be downloaded [here](). And the dataset ***Flickr15k*** can be downloaded [here](https://drive.google.com/open?id=13AFiwNh4FMks_jGfL4UDntMf0lHL6BTQ).
Canny edge detection procedure should be carried out to produce images' edge maps.

Second, you should modify the root path to ***Flickr15k*** at ``./train.py``.
The output of the model will be stored at ``./out/flickr15k_yymmddHHMM/*.pth``.
The default root path is ``../deep_hashing`` according to my case.

Third, run ``./train.py`` to train the network. Use ``./extract_feat_sketch.py`` and ``./extract_feat_photo.py`` to extract features from sketches and photograps.
The features will be stored at ``./out_feat/flickr15k_yymmddHHMM/feat_sketch.npz`` and ``./out_feat/flickr15k_yymmddHHMM/feat_photo.npz``.

Last, use ``./retrieval.py`` to gain results. The retrieval list will be stored at ``./out_feat/flickr15k_yymmddHHMM/result``. 
To be consistent with ***330sketches*** query's file structure, results of every query are saved in group and sorted by similariy. 

## Code Structure
```
.
├── accessory
│   └── pr_curve.png
├── dataset
│   ├── 330sketches
│   └── Flickr_15K_edge2
├── extract_feat_photo.py
├── extract_feat_sketch.py
├── flickr15k_dataset.py
├── model
│   ├── SketchTriplet_half_sharing.py
│   └── SketchTriplet.py
├── out
│   └── flickr15k_1904041458
│       ├── 500.pth
│       └── loss_and_accurary.txt
├── out_feat
│   └── flickr15k_1904041458
│       ├── feat_photo.npz
│       ├── feat_sketch.npz
│       └── result
├── README.md
├── retrieval.py
├── train.py
└── utils.py

```

## Results - Flickr15k

We will train the network ***SketchTriplet*** on the dataset ***Flickr15k***. 
The network takes an anchor (sketch input), positive (a photograph edgemap of same class as an anchor) and negative (photograph edgemaps of different class than an anchor) examples.

Some Parameters are shown as follows:

- Edge extraction algorithm: Canny edge detection
- Batch size: 128
- Number of epochs: 500
- Optimizer: torch.optim.SGD
    - learning rate: 1e-3
    - weight decay: 0.0005
    - momentum: 0.9
- Loss function: torch.nn.TripletMarginLoss
    - margin: 1.0
    - p: 2.0
    
After 500 epochs of training, here are the pr curve we get for testing set.

<p align="center">
    <img src="https://github.com/jjkislele/SketchTriplet/blob/master/accessory/pr_curve.png" width="300" height="300">
    <p align="center">
        <em>Pr curve for testing</em>
    </p>
</p>

Also the loss curve during training is shown as follows.

<p align="center">
    <img src="https://github.com/jjkislele/SketchTriplet/blob/master/accessory/loss.png" width="300" height="300">
    <p align="center">
        <em>Loss curve during training</em>
    </p>
</p>

Although it scores 67.7% mAP indicating just-so-so performance, the pr curve shows the model is over-fitting.

## Todo List
* [ ] Modified triplet loss function mentioned by the paper
* [ ] Eval process. 75% for training, 25% for evaluation at least
* [ ] Some silly code need be removed, refactoring also
* [ ] Extract features parallel
* [x] Fix typos

# Reference and Special Thanks

[1] adambielski's siamese-triplet [repo](https://github.com/adambielski/siamese-triplet)

[2] weixu000's DSH-pytorch [repo](https://github.com/weixu000/DSH-pytorch)

[3] TuBui's Triplet_Loss_SBIR [repo](https://github.com/TuBui/Triplet_Loss_SBIR)