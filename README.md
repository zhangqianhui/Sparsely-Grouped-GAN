# SG_GAN_tensorflow
The code of paper "Sparsely Grouped Multi-task Generative Adversarial Networks for Facial Attribute Manipulation"

## Paper
[Sparsely Grouped Multi-task Generative Adversarial Networks for Facial Attribute Manipulation](https://arxiv.org/abs/1805.07509) 

Introduction: Recently, Image-to-Image Translation (IIT) has made great progress in enabling image style transfer and manipulation of semantic context in an image. However, existing approaches require exhaustive labelling of training data, which is labor demanding, difficult to scale up, and hard to adapt to a new domain. To overcome such a key limitation, we propose sparsely grouped generative adversarial networks(SG-GAN), a novel approach that can perform image translation in the sparsely grouped datasets, which most training data are mixed and just a few are labelled. SG-GAN with one-input multiple output architecture can be used for the translations among multiple groups using only a single trained model. As a case study for experimentally validating the advantages of our model, we apply the algorithm to tackle a series of tasks of attribute manipulation for facial images. Experiment results show that SG-GAN can achieve competitive results compared with previous state-of-the-art methods on adequately labelled datasets while attaining the superior quality of image translation results on sparsely grouped datasets where most data is mixed and only small parts are labelled. 

<p align="center"><img width="100%" src="img/intro.JPG" /></p>

## Dependencies
* [Python 2.7](https://www.python.org/download/releases/2.7/)
* [Tensorflow 1.4+](https://github.com/tensorflow/tensorflow)

<br/>

## Usage

### 1. Cloning the repository
```bash
$ git clone https://github.com/yunjey/StarGAN.git
$ cd StarGAN/
```

### 2. Downloading the dataset
To download the CelebA dataset:
```bash
$ bash download.sh celeba
```

To download the RaFD dataset, you must request access to the dataset from [the Radboud Faces Database website](http://www.socsci.ru.nl:8180/RaFD2/RaFD?p=main). Then, you need to create a folder structure as described [here](https://github.com/yunjey/StarGAN/blob/master/jpg/RaFD.md).

### 3. Training
To train StarGAN on CelebA, run the training script below. See [here](https://github.com/yunjey/StarGAN/blob/master/jpg/CelebA.md) for a list of selectable attributes in the CelebA dataset. If you change the `selected_attrs` argument, you should also change the `c_dim` argument accordingly.

## Experiment Result 

### For balanced Attributes
<p align="center"><img width="100%" src="img/exper_1.PNG" /></p>
### For unbalanced Attribute
<p align="center"><img width="100%" src="img/exper_2.PNG" /></p>

## Reference code

[StarGAN Pytorch](https://github.com/yunjey/StarGAN)
[GeneGAN tensorflow](https://github.com/Prinsphield/GeneGAN)
[DCGAN tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)
[Spectral Norm tensorflow](https://github.com/taki0112/Spectral_Normalization-Tensorflow)
