from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.functional import pil_to_tensor
from torchvision import models
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings("ignore")

classes = ('Glass','Paper','Cardboard','Plastic','Metal','Trash')

class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('imgs/icon.png'))
        self.setWindowTitle('Garbage Classifiation System')
        self.net = models.resnet50(pretrained=False)
        self.net.fc = nn.Linear(in_features=2048, out_features=len(classes))
        self.net.load_state_dict(torch.load('cifar_net.pth', map_location=torch.device('cpu') ))
        self.net.eval()
        self.transform = v2.Compose([
            v2.Resize(size=(256, 256), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.resize(900, 600)
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        font = QFont('楷體', 20)
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        img_title = QLabel("Your Image")
        img_title.setFont(font)
        img_title.setAlignment(Qt.AlignCenter)
        self.img_label = QLabel()
        img_init = np.zeros((400, 400, 1), dtype = "uint8")
        cv2.imwrite('imgs/target.png', img_init)
        self.img_label.setPixmap(QPixmap('imgs/target.png'))
        left_layout.addWidget(img_title)
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        left_widget.setLayout(left_layout)

        right_widget = QWidget()
        right_layout = QVBoxLayout()
        btn_change = QPushButton(" Upload Image ")
        btn_change.clicked.connect(self.change_img)
        btn_change.setFont(font)

        label_result = QLabel('Result:')
        self.result = QLabel("(Unknown)")
        label_result.setFont(QFont('楷體', 20))
        self.result.setFont(QFont('楷體', 32))
        right_layout.addStretch()
        right_layout.addWidget(label_result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(btn_change)
        right_layout.addStretch()
        right_widget.setLayout(right_layout)

        # 关于页面
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('Welcome to Kaggle Garbage Classification System!')
        about_title.setFont(QFont('楷體', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('imgs/classes.png'))
        about_img.setAlignment(Qt.AlignCenter)
        # git_img = QMovie('images/')
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        about_widget.setLayout(about_layout)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_widget.setLayout(main_layout)
        self.addTab(main_widget, 'Home')
        self.addTab(about_widget, 'About')
        self.setTabIcon(0, QIcon('imgs/icon.png'))
        self.setTabIcon(1, QIcon('imgs/qicon.png'))

    def change_img(self):
        openfile_name = QFileDialog.getOpenFileName(self, 'Select File', '', 'Image files(*.jpg *.png *.jpeg)')
        print(openfile_name)
        img_name = openfile_name[0]
        if img_name == '':
            pass
        else:
            self.predict_img_path = img_name
            img_init = cv2.imread(self.predict_img_path)
            img_init = cv2.resize(img_init, (400, 400))
            cv2.imwrite('imgs/target.png', img_init)
            self.img_label.setPixmap(QPixmap('imgs/target.png'))
            self.predict_img()

    def predict_img(self):
        # 预测图片
        # 开始预测
        # img = Image.open()
        transform = v2.Compose([
            v2.Resize(size=(256, 256), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img = Image.open(self.predict_img_path)
        RGB_img = img.convert('RGB')
        img_torch = pil_to_tensor(RGB_img)
        img_torch = transform(img_torch)
        img_torch = img_torch.view(-1, 3, 256, 256)
        outputs = self.net(img_torch)
        _, predicted = torch.max(outputs, 1)
        result = str(classes[predicted[0].numpy()])

        self.result.setText(result)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    x = MainWindow()
    x.show()
    sys.exit(app.exec_())