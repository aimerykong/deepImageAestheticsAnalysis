# Photo Aesthetics Ranking Network with Attributes and Content Adaptation
### Code, demo and model for our project of deep image aesthetics analysis

We are releasing our codes/demo/dataset used in our project of image aesthetics analysis. This project is jointed done in Adobe Research and UCI. A patent is filed.

![alt text](http://www.ics.uci.edu/~skong2/img/aestheticsDemoFigure.png "display")

The AADB dataset is large (datasetImages.zip, 2.0GB), so we attach it to [google drive](https://drive.google.com/open?id=0BxeylfSgpk1MOVduWGxyVlJFUHM) from where a smaller version of AADB can also be downloaded with resized images (256x256 pixel resolution, datasetImages_warp256.zip, 130MB). Please note that all the images are downloaded from flickr with Creative Commons license, so the dataset is for research purpose only.

Technically, the rank loss is implemented in caffe. The modified caffe (named "caffeCustom.zip") can be downloaded in the google drive. An example prototxt to train the model can also be found there, named "mergedNetRank.prototxt". 

Some models are also released along with a demo interface. Running the demo can give you a clear way on how to load/interpret the model. The model can be downloaded from the google drive as well. As well, the models are for research purpose only as patent has been filed by Adobe.

Besides, a model trained on AVA dataset is released and stored in google drive, named "AVA_modelRelease.zip". There are matlab code pieces to test the model. Note that the matcaffe path might need to change, but it's trivial.

For further questions, please refer to our ECCV2016 paper or send me email through (skong2 AT uci DOT edu)

If you find our model/method/dataset useful, please cite our work:

    @inproceedings{kong2016aesthetics,
      title={Photo Aesthetics Ranking Network with Attributes and Content Adaptation},
      author={Kong, Shu and Shen, Xiaohui and Lin, Zhe and Mech, Radomir and Fowlkes, Charless},
      booktitle={ECCV},
      year={2016}
    }

Latest edit
Mar. 21, 2017


