# Garbage Classifiation System From Kaggle
This is a simple garbage classifier trained with resnet50 using pytorch with above 90% accuracy. The result could be 'glass','paper','cardboard','plastic','metal',and 'trash'.

## Windows Usage
Download the zip folder first.<br>
Unzip the folder, double click `main.py` and run the app.<br>
If you wanna know how to train the model, the whole training process is written in `train.ipynb`. You can train model on your own.


## Linux Usage
Make sure you have **python and pip** install in your system.
```sh
sudo apt install python
sudo apt install python-pip
```
Clone the repository and change directory.
```sh
git clone https://github.com/cccbeginner/KaggleGarbageClassifier.git
cd KaggleGarbageClassifier
```
Ensure you have necessary python packages installed.
```sh
$ pip install -r requirements.txt
```

Finally, you are able to run my project.
```
python main.py
```

## Training process
If you run the project directly, the model you use is already trained. If you wanna know how to train the model, you may view the whole training process in `train.ipynb`. The repo also includes `GarbagePictures.zip`, which contains dataset downloaded from original kaggle contest site. You may train and improve the model as well.

## Reference
- [Link to original kaggle contest](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
- [Link to the UI template](https://blog.csdn.net/ECHOSON/article/details/114396159)