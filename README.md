# Photo Aesthetics Ranking Network with Attributes and Content Adaptation
### Code, demo and model for our project of [deep image aesthetics analysis](https://www.ics.uci.edu/~skong2/aesthetics.html)

We are releasing our codes/demo/dataset used in our project of image aesthetics analysis. This project is jointed done in Adobe Research and UCI. Note that the patent [US20170294010A1](https://patents.google.com/patent/US20170294010A1/en) discourages considerations of  commercial use.

![alt text](http://www.ics.uci.edu/~skong2/img/aestheticsDemoFigure.png "display")

The AADB dataset is large, so we attach it to [**google drive**](https://drive.google.com/open?id=0BxeylfSgpk1MOVduWGxyVlJFUHM) from where a smaller version of AADB can also be downloaded with resized images (256x256 pixel resolution, [datasetImages_warp256.zip](https://drive.google.com/open?id=0BxeylfSgpk1MU2RsVXo3bEJWM2c), 130MB). For full-resolution training set (2GB), please download [this](https://drive.google.com/open?id=1Viswtzb77vqqaaICAQz9iuZ8OEYCu6-_); full-resolution testing set (200MB), [here](https://drive.google.com/open?id=115qnIQ-9pl5Vt06RyFue3b6DabakATmJ).
Please note that all the images are downloaded from flickr with Creative Commons license, so the dataset is for research purpose only.

Technically, the rank loss is implemented in caffe. The modified caffe (named "[caffeCustom.zip](https://drive.google.com/open?id=0BxeylfSgpk1MVXM2clpjeDhVRms)") can be downloaded in the google drive. An example prototxt to train the model can also be found there, named "mergedNetRank.prototxt". 

Some models are also released along with a demo interface. Running the demo can give you a clear way on how to load/interpret the model. The model can be downloaded from the google drive as well. As well, the models are for research purpose only as patent has been filed by Adobe.

Besides, a model trained on AVA dataset is released and stored in google drive, named "[AVA_modelRelease.zip](https://drive.google.com/open?id=0BxeylfSgpk1Mb2pwZlFwRlRmekk)". There are matlab code pieces to test the model. Note that the matcaffe path might need to change, but it's trivial.

For further questions, please refer to our ECCV2016 paper or send me email through (skong2 AT uci DOT edu)

If you find our model/method/dataset useful, please cite our work:

    @inproceedings{kong2016aesthetics,
      title={Photo Aesthetics Ranking Network with Attributes and Content Adaptation},
      author={Kong, Shu and Shen, Xiaohui and Lin, Zhe and Mech, Radomir and Fowlkes, Charless},
      booktitle={ECCV},
      year={2016}
    }

created: Mar. 21, 2017
last edited: May 29, 2019

Shu Kong @ UCI
