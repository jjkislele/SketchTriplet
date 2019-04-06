# Triplet_Loss_SBIR in Pytorch
This repo contains code implemented by Pytorch for the T.Bui et al's paper "[Compact Descriptors for Sketch-based Image Retrieval using a Triplet loss Convolutional Neural Network](https://doi.org/10.1016/j.cviu.2017.06.007)"[[Repo](https://github.com/TuBui/Triplet_Loss_SBIR)|[Page](http://www.cvssp.org/data/Flickr25K/CVIU16.html)].

The difference of the perposed network's architecture confuses me. In the paper, shown as Figure. 1, in each branch, the ``conv4`` layer don't have ``ReLu`` node right behind it, though, in [original codes](https://github.com/TuBui/Triplet_Loss_SBIR/blob/master/models/train.prototxt) ``conv4`` does.
I consult the [original codes](https://github.com/TuBui/Triplet_Loss_SBIR/blob/master/models/train.prototxt) to build the net.

## Installation

## How to Run

## Code Structure

## Todo List
* [] Modified triplet loss function mentioned by the paper
* [] Eval process 
* [] Some silly code

# Reference and Special Thanks

[1] adambielski's siamese-triplet [repo]()

[2] weixu000's DSH-pytorch [repo]()

[3] TuBui's Triplet_Loss_SBIR [repo]()