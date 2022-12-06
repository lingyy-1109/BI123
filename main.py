import numpy

from BMEUI.BI123 import Ui_MainWindow # 导入UI文件
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import numpy as np
import cv2
import pyqtgraph as pg
import math
from scipy import signal
from skimage import morphology
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
        self.ConfirmButton.clicked.connect(self.Confirm)
        self.SobelButton.clicked.connect(self.SobelFilter)
        self.MedianButton.clicked.connect(self.MedianFilter)
        self.DilationButton.clicked.connect(self.Dilation)
        self.ErosionButton.clicked.connect(self.Erosion)
        self.OpeningButton.clicked.connect(self.Opening)
        self.ClosingButton.clicked.connect(self.Closing)
        self.DistanceTransformButton.clicked.connect(self.DistanceTransform)
        self.SkeletonButton.clicked.connect(self.Skeleton)
        self.SkeletonRestorationButton.clicked.connect(self.SkeletonRestoration)
        self.actionExport_Image.triggered.connect(self.export_img)

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

    # 图像+直方图绘制
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
        # 清空画布
        pw.clear()
        # 显示数据
        if img.dtype == 'uint8':
            weight = 256
        elif img.dtype == 'uint16':
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

        else:
            hist = cv2.calcHist([img], [0], None, [weight], [0, weight])
            hist_b = np.ravel(hist[1:])

            penstate = pg.mkPen(width=2, color='c')  # 绘图格式

            pw.plot(x, hist_b, stepMode="center", pen=penstate)

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


    # 打开、保存图像
    def open_jpg(self):
        filePath, filetype = QFileDialog.getOpenFileName(self, "选取图像", "./", "JPG Image(*.jpg)")
        self.originalImg = cv2.imread(filePath)
        self.print_img(self.originalImg, self.OriImg, self.pwOri)

    def open_bmp(self):
        filePath, filetype = QFileDialog.getOpenFileName(self, "选取图像", "./", "BMP Image(*.bmp)")
        self.originalImg = cv2.imread(filePath)
        self.print_img(self.originalImg, self.OriImg, self.pwOri)

    def export_img(self):
        cv2.imwrite('ExportImg.jpg', self.originalImg)
    # 确认操作(将处理后的图像保存为处理前的图像)
    def Confirm(self):
        if (self.originalImg is not None and self.processedImg is not None):
            self.originalImg = self.processedImg
            self.print_img(self.originalImg, self.OriImg, self.pwOri)

    # Lab1:图像分割
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
                p.insert(i, hist[i][0] / gray.size)
            for k in range(256):
                H_b = 0  # black的熵，前景的平均信息量
                H_w = 0  # white的熵，背景的平均信息量
                for i in range(k):
                    if p[i] != 0:
                        H_b = H_b - p[i] * math.log(10, p[i])

                for i in range(k, 256):
                    if p[i] != 0:
                        H_w = H_w - p[i] * math.log(10, p[i])

                H = H_b + H_w
                if H > H_last:
                    H_last = H
                    best_k = k
            # print(best_k)
            ret, self.processedImg = cv2.threshold(gray, best_k, 255, cv2.THRESH_BINARY)
            # self.processedImg = gray
            self.print_img(self.processedImg, self.ProImg, self.pwPro)

    # Lab2:图像滤波
    def SobelFilter(self):
        if(self.originalImg is not None):
            # 首先应当存在初始图像
            if(len(self.originalImg.shape) == 3):
                gray = cv2.cvtColor(self.originalImg, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.originalImg

            filter = np.array([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ])
            self.processedImg = signal.convolve2d(gray, filter, mode='same', boundary='symm').astype(gray.dtype)
            print(self.processedImg.dtype)
            self.print_img(self.processedImg, self.ProImg, self.pwPro)

    def MedianFilter(self):
        if(self.originalImg is not None):
            # 首先应当存在初始图像
            if(len(self.originalImg.shape) == 3):
                gray = cv2.cvtColor(self.originalImg, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.originalImg

            filter = np.array([
                [1/9, 1/9, 1/9],
                [1/9, 1/9, 1/9],
                [1/9, 1/9, 1/9]
            ])
            self.processedImg = signal.convolve2d(gray, filter, mode='same', boundary='symm').astype(gray.dtype)
            print(self.processedImg.dtype)
            self.print_img(self.processedImg, self.ProImg, self.pwPro)

    # Lab3:形态学处理 手写dilation//erosion
    def Dilation(self):
        # 首先应当存在初始图像
        if(self.originalImg is not None):
            # 判断图像为彩色图像还是灰度图像
            if(len(self.originalImg.shape) == 3):
                gray = cv2.cvtColor(self.originalImg, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.originalImg

            # 系统处理的图像不全是二值图像,需要先转换为二值图像
            binaryImg = gray.copy()
            finalImg = gray.copy()
            binaryImg[gray >= 128] = 255
            binaryImg[gray < 128] = 0
            numpy.pad(binaryImg, [(1, 1), (1, 1)], 'constant', constant_values = [(255, 255), (255, 255)])
            kernal = np.array([
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ])
            height, width = binaryImg.shape
            for i in range(height - 2):
                for j in range(width - 2):
                    slice = binaryImg[i:i+3, j:j+3]
                    jud = kernal * slice
                    if ((jud - kernal * 255).astype(gray.dtype).any()):
                        finalImg[i][j] = 0
                    else:
                        finalImg[i][j] = 255
            self.processedImg = finalImg.astype(gray.dtype)
            self.print_img(self.processedImg, self.ProImg, self.pwPro)

    def Erosion(self):
        # 首先应当存在初始图像
        if(self.originalImg is not None):
            # 判断图像为彩色图像还是灰度图像
            if(len(self.originalImg.shape) == 3):
                gray = cv2.cvtColor(self.originalImg, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.originalImg

            # 系统处理的图像不全是二值图像,需要先转换为二值图像
            binaryImg = gray.copy()
            finalImg = gray.copy()
            binaryImg[gray >= 128] = 255
            binaryImg[gray < 128] = 0
            numpy.pad(binaryImg, [(1, 1), (1, 1)], 'constant')
            kernal = np.array([
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ])
            height, width = binaryImg.shape
            for i in range(height - 2):
                for j in range(width - 2):
                    slice = binaryImg[i:i+3, j:j+3]
                    jud = kernal * slice
                    if ((jud).astype(gray.dtype).any()):
                        finalImg[i][j] = 255
                    else:
                        finalImg[i][j] = 0
            self.processedImg = finalImg.astype(gray.dtype)
            self.print_img(self.processedImg, self.ProImg, self.pwPro)

    def Opening(self):
        # 首先应当存在初始图像
        if(self.originalImg is not None):
            # 判断图像为彩色图像还是灰度图像
            if(len(self.originalImg.shape) == 3):
                gray = cv2.cvtColor(self.originalImg, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.originalImg

            # 系统处理的图像不全是二值图像,需要先转换为二值图像
            binaryImg = gray.copy()
            tempImg = gray.copy()
            finalImg = gray.copy()
            binaryImg[gray >= 128] = 255
            binaryImg[gray < 128] = 0
            kernal = np.array([
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ])
            # Dilation
            numpy.pad(binaryImg, [(1, 1), (1, 1)], 'constant', constant_values=[(255, 255),(255, 255)])
            height, width = binaryImg.shape
            for i in range(height - 2):
                for j in range(width - 2):
                    slice = binaryImg[i:i+3, j:j+3]
                    jud = kernal * slice
                    if ((jud - kernal * 255).astype(gray.dtype).any()):
                        tempImg[i][j] = 0
                    else:
                        tempImg[i][j] = 255
            # Erosion
            numpy.pad(tempImg, [(1, 1), (1, 1)], 'constant')
            height, width = tempImg.shape
            for i in range(height - 2):
                for j in range(width - 2):
                    slice = tempImg[i:i+3, j:j+3]
                    jud = kernal * slice
                    if ((jud).astype(gray.dtype).any()):
                        finalImg[i][j] = 255
                    else:
                        finalImg[i][j] = 0
            self.processedImg = finalImg.astype(gray.dtype)
            self.print_img(self.processedImg, self.ProImg, self.pwPro)

    def Closing(self):
        # 首先应当存在初始图像
        if(self.originalImg is not None):
            # 判断图像为彩色图像还是灰度图像
            if(len(self.originalImg.shape) == 3):
                gray = cv2.cvtColor(self.originalImg, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.originalImg

            # 系统处理的图像不全是二值图像,需要先转换为二值图像
            binaryImg = gray.copy()
            tempImg = gray.copy()
            finalImg = gray.copy()
            binaryImg[gray >= 128] = 255
            binaryImg[gray < 128] = 0
            kernal = np.array([
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ])

            # Erosion
            numpy.pad(binaryImg, [(1, 1), (1, 1)], 'constant')
            height, width = binaryImg.shape
            for i in range(height - 2):
                for j in range(width - 2):
                    slice = binaryImg[i:i+3, j:j+3]
                    jud = kernal * slice
                    if ((jud).astype(gray.dtype).any()):
                        tempImg[i][j] = 255
                    else:
                        tempImg[i][j] = 0
            # Dilation
            numpy.pad(tempImg, [(1, 1), (1, 1)], 'constant',constant_values=[(255, 255),(255, 255)])
            height, width = tempImg.shape
            for i in range(height - 2):
                for j in range(width - 2):
                    slice = tempImg[i:i+3, j:j+3]
                    jud = kernal * slice
                    if ((jud - kernal * 255).astype(gray.dtype).any()):
                        finalImg[i][j] = 0
                    else:
                        finalImg[i][j] = 255
            self.processedImg = finalImg.astype(gray.dtype)
            self.print_img(self.processedImg, self.ProImg, self.pwPro)

    # Lab4:其他处理算法
    def DistanceTransform(self):
        # 首先应当存在初始图像
        if(self.originalImg is not None):
            # 判断图像为彩色图像还是灰度图像
            if(len(self.originalImg.shape) == 3):
                gray = cv2.cvtColor(self.originalImg, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.originalImg

            #二值化处理
            binaryImg = gray.copy()
            binaryImg[gray >= 128] = 1
            binaryImg[gray < 128] = 0
            finalImg = binaryImg.copy()

            kernal = np.array([
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ])

            #迭代求距离
            print(np.all(binaryImg != 0))
            while not np.all(binaryImg == 0):
                # Dilation

                numpy.pad(binaryImg, [(1, 1), (1, 1)], 'constant', constant_values=[(255, 255), (255, 255)])
                height, width = binaryImg.shape
                binaryImg = cv2.erode(binaryImg, kernal, iterations=1)
                finalImg = finalImg + binaryImg
            self.processedImg = finalImg.astype(gray.dtype)

            # 在输出的时候映射阈值
            ImageForPrint = finalImg  / np.max(finalImg) * 255
            ImageForPrint = ImageForPrint.astype(gray.dtype)

            self.print_img(ImageForPrint, self.ProImg, self.pwPro)

    def Skeleton(self):
        # 首先应当存在初始图像
        if(self.originalImg is not None):
            # 判断图像为彩色图像还是灰度图像
            if(len(self.originalImg.shape) == 3):
                gray = cv2.cvtColor(self.originalImg, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.originalImg

            ret, imgBin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)  # 二值化处理

            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            skeleton = np.zeros(imgBin.shape, np.uint8)  # 创建空骨架图
            while True:
                imgOpen = cv2.morphologyEx(imgBin, cv2.MORPH_OPEN, element)  # 开运算
                subSkel = cv2.subtract(imgBin, imgOpen)  # 获得骨架子集
                skeleton = cv2.bitwise_or(skeleton, subSkel)  # # 将删除的像素添加到骨架图
                imgBin = cv2.erode(imgBin, element)  # 腐蚀，用于下一次迭代
                if cv2.countNonZero(imgBin) == 0:
                    break

            self.processedImg = skeleton * 255 * 255
            self.print_img(self.processedImg, self.ProImg, self.pwPro)

    def SkeletonRestoration(self):
        # 首先应当存在初始图像
        if(self.originalImg is not None):
            # 判断图像为彩色图像还是灰度图像
            if(len(self.originalImg.shape) == 3):
                gray = cv2.cvtColor(self.originalImg, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.originalImg

            ret, imgBin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)  # 二值化处理

            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            skeleton = np.zeros(imgBin.shape, np.uint8)  # 创建空骨架图
            skeletonRestoration = np.zeros(imgBin.shape, np.uint8)
            subSkelLine = [] # 用于存放S_i
            while True:
                imgOpen = cv2.morphologyEx(imgBin, cv2.MORPH_OPEN, element)  # 开运算
                subSkel = cv2.subtract(imgBin, imgOpen)  # 获得骨架子集
                subSkelLine.append(subSkel)
                skeleton = cv2.bitwise_or(skeleton, subSkel)  # # 将删除的像素添加到骨架图
                imgBin = cv2.erode(imgBin, element)  # 腐蚀，用于下一次迭代
                if cv2.countNonZero(imgBin) == 0:
                    break

            i = 0
            for oneSubSkel in subSkelLine:
                i += 1
                imgDilation = cv2.dilate(oneSubSkel, element, iterations= i)
                skeletonRestoration = cv2.bitwise_or(skeletonRestoration, imgDilation)

            self.processedImg = skeletonRestoration * 255 * 255
            self.print_img(self.processedImg, self.ProImg, self.pwPro)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    mywindow = Window()
    mywindow.show()
    sys.exit(app.exec_())



