from BMEUI.BI123 import Ui_MainWindow # 导入UI文件
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import numpy as np
from PIL import Image
import cv2
import matplotlib
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import math

class Window(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)
        self.initHist()
        self.originalImg = None
        self.processedImg = None
        self.actionImport_jpg.triggered.connect(self.open_jpg)
        self.actionImport_bmp.triggered.connect(self.open_bmp)
        self.OtsuButton.clicked.connect(self.Otsu)
        self.EntropyButton.clicked.connect(self.Entropy)




    def initHist(self):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.pwOri = pg.PlotWidget(self.OriHis, enableMenu=False)  # 创建一个绘图控件
        self.pwOri.resize(720, 300)
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.pwOri.addItem(self.vLine, ignoreBounds=True)
        self.pwOri.addItem(self.hLine, ignoreBounds=True)
        proxy1 = pg.SignalProxy(self.pwOri.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMovedOri)  # 鼠标移动获取位置

        self.pwPro = pg.PlotWidget(self.ProHis, enableMenu=False)  # 创建一个绘图控件
        self.pwPro.resize(720, 300)
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.pwPro.addItem(self.vLine, ignoreBounds=True)
        self.pwPro.addItem(self.hLine, ignoreBounds=True)
        proxy2 = pg.SignalProxy(self.pwPro.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMovedPro)  # 鼠标移动获取位置

    def mouseMovedOri(self, evt):
        pos = evt[0]
        if self.pwOri.sceneBoundingRect().contains(pos):
            mousePoint = self.pwOri.vb.mapSceneToView(pos)
            index = int(mousePoint.x())
            print(index)
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())

    def mouseMovedPro(self, evt):
        pos = evt[0]
        if self.pwPro.sceneBoundingRect().contains(pos):
            mousePoint = self.pwPro.vb.mapSceneToView(pos)
            index = int(mousePoint.x())
            print(index)
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())

    def drawHist(self, img, pw):  # 绘制图像直方图
        # 显示数据
        rows, cols = img.shape[:2]

        if img.dtype == 'uint8':
            deep = '8位'
            weight = 256
        elif img.dtype == 'uint16':
            deep = '16位'
            weight = 65536

        x = np.linspace(0.5, weight - 0.5, weight)

        if len(img.shape) == 3:
            # self.label_imgcolor.setText("彩色")

            hist_b = cv2.calcHist([img], [0], None, [weight], [0, weight])
            hist_b = np.ravel(hist_b[1:])
            hist_g = cv2.calcHist([img], [1], None, [weight], [0, weight])
            hist_g = np.ravel(hist_g[1:])
            hist_r = cv2.calcHist([img], [2], None, [weight], [0, weight])
            hist_r = np.ravel(hist_r[1:])
            penstate_b = pg.mkPen(width=2, color='b')  # 绘图格式
            penstate_g = pg.mkPen(width=2, color='g')  # 绘图格式
            penstate_r = pg.mkPen(width=2, color='r')  # 绘图格式
            pw.plot(x, hist_b, stepMode="center", pen=penstate_b)
            pw.plot(x, hist_g, stepMode="center", pen=penstate_g)
            pw.plot(x, hist_r, stepMode="center", pen=penstate_r)

            # self.label_imgdeep.setText(str(deep))
            minVal, maxVal, _, _ = cv2.minMaxLoc(img[:, :, 0])
            # self.label_max.setText(str(int(maxVal)))
            # self.label_min.setText(str(int(minVal)))
            mean, std = cv2.meanStdDev(img[:, :, 0])
            mean = round(mean[0][0], 3)
            std = round(std[0][0], 3)
            # self.label_mean.setText(str(mean))
            # self.label_stddev.setText(str(std))
        else:
            # self.label_imgcolor.setText("黑白")

            hist = cv2.calcHist([img], [0], None, [weight], [0, weight])
            hist_b = np.ravel(hist[1:])

            penstate = pg.mkPen(width=2, color='c')  # 绘图格式

            pw.plot(x, hist_b, stepMode="center", pen=penstate)

            # self.label_imgdeep.setText(str(deep))
            minVal, maxVal, _, _ = cv2.minMaxLoc(img)
            # self.label_max.setText(str(int(maxVal)))
            # self.label_min.setText(str(int(minVal)))
            mean, std = cv2.meanStdDev(img)
            mean = round(mean[0][0], 3)
            std = round(std[0][0], 3)
            # self.label_mean.setText(str(mean))
            # self.label_stddev.setText(str(std))

    def print_img(self, image, show_label, hist_pw):
        # 在图像模块显示原图 分彩色图像和黑白图像讨论
        if(len(image.shape) == 3):
            RGBImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            QTImg = QImage(RGBImg, RGBImg.shape[1], RGBImg.shape[0], RGBImg.shape[1] * 3, QImage.Format_RGB888)
            QTImg = QPixmap(QTImg)
            show_label.setPixmap(QTImg)
        else:
            RGBImg = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            QTImg = QImage(RGBImg, RGBImg.shape[1], RGBImg.shape[0], RGBImg.shape[1] * 3, QImage.Format_RGB888)
            QTImg = QPixmap(QTImg)
            show_label.setPixmap(QTImg)

        # 绘制灰度直方图
        self.drawHist(image, hist_pw)

    def open_jpg(self):
        filePath, filetype = QFileDialog.getOpenFileName(self, "选取图像", "./", "JPG Image(*.jpg)")
        self.originalImg = cv2.imread(filePath)
        self.print_img(self.originalImg, self.OriImg, self.pwOri)

    def open_bmp(self):
        filePath, filetype = QFileDialog.getOpenFileName(self, "选取图像", "./", "BMP Image(*.bmp)")
        self.originalImg = cv2.imread(filePath)
        self.print_img(self.originalImg, self.OriImg, self.pwOri)

    def Otsu(self):
        if(self.originalImg is not None):
            # 首先应当存在初始图像
            if(len(self.originalImg.shape) == 3):
                gray = cv2.cvtColor(self.originalImg, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.originalImg
            pixel_number = gray.shape[0] * gray.shape[1]
            mean_weigth = 1.0 / pixel_number
            # 发现bins必须写到257，否则255这个值只能分到[254,255)区间
            his, bins = np.histogram(gray, np.arange(0, 257))
            final_thresh = -1
            final_value = -1
            intensity_arr = np.arange(256)
            for t in bins[1:-1]:  # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
                pcb = np.sum(his[:t])
                pcf = np.sum(his[t:])
                Wb = pcb * mean_weigth
                Wf = pcf * mean_weigth

                mub = np.sum(intensity_arr[:t] * his[:t]) / float(pcb)
                muf = np.sum(intensity_arr[t:] * his[t:]) / float(pcf)
                # print mub, muf
                value = Wb * Wf * (mub - muf) ** 2

                if value > final_value:
                    final_thresh = t
                    final_value = value
            final_img = gray.copy()
            print(final_thresh)
            final_img[gray > final_thresh] = 255
            final_img[gray < final_thresh] = 0
            self.processedImg = final_img
            self.print_img(self.processedImg, self.ProImg, self.pwPro)

    def Entropy(self):
        if(self.originalImg is not None):
            # 首先应当存在初始图像
            if(len(self.originalImg.shape) == 3):
                gray = cv2.cvtColor(self.originalImg, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.originalImg

            p = []  # 灰度概率
            H_last = 0  # 上一个H总熵
            best_k = 0  # 最佳阈值
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])  # 255*1的灰度直方图的数组
            for i in range(256):
                p.insert(i, hist[i][0] /    092.size)
            for k in range(256):
                H_b = 0  # black的熵，前景的平均信息量
                H_w = 0  # white的熵，背景的平均信息量
                for i in range(k):
                    if p[i] != 0:
                        H_b = H_b - p[i] * math.log(2, p[i])

                for i in range(k, 256):
                    if p[i] != 0:
                        H_w = H_w - p[i] * math.log(2, p[i])

                H = H_b + H_w
                if H > H_last:
                    H_last = H
                    best_k = k
            print(best_k)
            ret, self.processedImg = cv2.threshold(gray, best_k, 255, cv2.THRESH_BINARY)
            self.processedImg = gray
            self.print_img(self.processedImg, self.ProImg, self.pwPro)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mywindow = Window()
    mywindow.show()
    sys.exit(app.exec_())



