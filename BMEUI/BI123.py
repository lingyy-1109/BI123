# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'BI123.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1354, 857)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.OriImgName = QtWidgets.QLabel(self.centralwidget)
        self.OriImgName.setObjectName("OriImgName")
        self.horizontalLayout.addWidget(self.OriImgName)
        self.ProImgName = QtWidgets.QLabel(self.centralwidget)
        self.ProImgName.setObjectName("ProImgName")
        self.horizontalLayout.addWidget(self.ProImgName)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.OriWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.OriWidget.setObjectName("OriWidget")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.tab_3)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.OriImg = QtWidgets.QLabel(self.tab_3)
        self.OriImg.setObjectName("OriImg")
        self.verticalLayout_8.addWidget(self.OriImg)
        self.OriWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.tab_4)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.OriHis = QtWidgets.QWidget(self.tab_4)
        self.OriHis.setObjectName("OriHis")
        self.verticalLayout_5.addWidget(self.OriHis)
        self.OriWidget.addTab(self.tab_4, "")
        self.horizontalLayout_3.addWidget(self.OriWidget)
        self.ProWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.ProWidget.setObjectName("ProWidget")
        self.tab_7 = QtWidgets.QWidget()
        self.tab_7.setObjectName("tab_7")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.tab_7)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.ProImg = QtWidgets.QLabel(self.tab_7)
        self.ProImg.setText("")
        self.ProImg.setObjectName("ProImg")
        self.verticalLayout_6.addWidget(self.ProImg)
        self.ProWidget.addTab(self.tab_7, "")
        self.tab_8 = QtWidgets.QWidget()
        self.tab_8.setObjectName("tab_8")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.tab_8)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.ProHis = QtWidgets.QWidget(self.tab_8)
        self.ProHis.setObjectName("ProHis")
        self.verticalLayout_7.addWidget(self.ProHis)
        self.ProWidget.addTab(self.tab_8, "")
        self.horizontalLayout_3.addWidget(self.ProWidget)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.ConfirmButton = QtWidgets.QPushButton(self.centralwidget)
        self.ConfirmButton.setObjectName("ConfirmButton")
        self.gridLayout_2.addWidget(self.ConfirmButton, 0, 1, 1, 1)
        self.FunctionWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.FunctionWidget.setObjectName("FunctionWidget")
        self.tab_9 = QtWidgets.QWidget()
        self.tab_9.setObjectName("tab_9")
        self.gridLayout = QtWidgets.QGridLayout(self.tab_9)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem, 2, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem1, 0, 1, 1, 1)
        self.EntropyButton = QtWidgets.QPushButton(self.tab_9)
        self.EntropyButton.setObjectName("EntropyButton")
        self.gridLayout_3.addWidget(self.EntropyButton, 4, 1, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem2, 1, 2, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem3, 1, 0, 1, 1)
        self.OtsuButton = QtWidgets.QPushButton(self.tab_9)
        self.OtsuButton.setObjectName("OtsuButton")
        self.gridLayout_3.addWidget(self.OtsuButton, 1, 1, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem4, 5, 1, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        self.FunctionWidget.addTab(self.tab_9, "")
        self.tab_11 = QtWidgets.QWidget()
        self.tab_11.setObjectName("tab_11")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tab_11)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_7 = QtWidgets.QGridLayout()
        self.gridLayout_7.setObjectName("gridLayout_7")
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_7.addItem(spacerItem5, 1, 2, 1, 1)
        self.SobelButton = QtWidgets.QPushButton(self.tab_11)
        self.SobelButton.setObjectName("SobelButton")
        self.gridLayout_7.addWidget(self.SobelButton, 1, 1, 1, 1)
        self.MedianButton = QtWidgets.QPushButton(self.tab_11)
        self.MedianButton.setObjectName("MedianButton")
        self.gridLayout_7.addWidget(self.MedianButton, 3, 1, 1, 1)
        spacerItem6 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_7.addItem(spacerItem6, 0, 1, 1, 1)
        spacerItem7 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_7.addItem(spacerItem7, 2, 1, 1, 1)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_7.addItem(spacerItem8, 1, 0, 1, 1)
        spacerItem9 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_7.addItem(spacerItem9, 4, 1, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_7, 0, 0, 1, 1)
        self.FunctionWidget.addTab(self.tab_11, "")
        self.tab_10 = QtWidgets.QWidget()
        self.tab_10.setObjectName("tab_10")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.tab_10)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.gridLayout_8 = QtWidgets.QGridLayout()
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.DilationButton = QtWidgets.QPushButton(self.tab_10)
        self.DilationButton.setObjectName("DilationButton")
        self.gridLayout_8.addWidget(self.DilationButton, 1, 1, 1, 1)
        self.DistanceTransformButton = QtWidgets.QPushButton(self.tab_10)
        self.DistanceTransformButton.setObjectName("DistanceTransformButton")
        self.gridLayout_8.addWidget(self.DistanceTransformButton, 1, 3, 1, 1)
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_8.addItem(spacerItem10, 1, 2, 1, 1)
        self.SkeletonRestorationButton = QtWidgets.QPushButton(self.tab_10)
        self.SkeletonRestorationButton.setObjectName("SkeletonRestorationButton")
        self.gridLayout_8.addWidget(self.SkeletonRestorationButton, 5, 3, 1, 1)
        spacerItem11 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_8.addItem(spacerItem11, 1, 0, 1, 1)
        spacerItem12 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_8.addItem(spacerItem12, 0, 1, 1, 1)
        self.ErosionButton = QtWidgets.QPushButton(self.tab_10)
        self.ErosionButton.setObjectName("ErosionButton")
        self.gridLayout_8.addWidget(self.ErosionButton, 3, 1, 1, 1)
        self.OpeningButton = QtWidgets.QPushButton(self.tab_10)
        self.OpeningButton.setObjectName("OpeningButton")
        self.gridLayout_8.addWidget(self.OpeningButton, 5, 1, 1, 1)
        self.ClosingButton = QtWidgets.QPushButton(self.tab_10)
        self.ClosingButton.setObjectName("ClosingButton")
        self.gridLayout_8.addWidget(self.ClosingButton, 7, 1, 1, 1)
        spacerItem13 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_8.addItem(spacerItem13, 6, 1, 1, 1)
        self.SkeletonButton = QtWidgets.QPushButton(self.tab_10)
        self.SkeletonButton.setObjectName("SkeletonButton")
        self.gridLayout_8.addWidget(self.SkeletonButton, 3, 3, 1, 1)
        spacerItem14 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_8.addItem(spacerItem14, 1, 4, 1, 1)
        spacerItem15 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_8.addItem(spacerItem15, 2, 1, 1, 1)
        spacerItem16 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_8.addItem(spacerItem16, 4, 1, 1, 1)
        spacerItem17 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_8.addItem(spacerItem17, 8, 1, 1, 1)
        self.EdgeDetectionButton = QtWidgets.QPushButton(self.tab_10)
        self.EdgeDetectionButton.setObjectName("EdgeDetectionButton")
        self.gridLayout_8.addWidget(self.EdgeDetectionButton, 7, 3, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_8, 0, 0, 1, 1)
        self.FunctionWidget.addTab(self.tab_10, "")
        self.tab_12 = QtWidgets.QWidget()
        self.tab_12.setObjectName("tab_12")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.tab_12)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.gridLayout_9 = QtWidgets.QGridLayout()
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.GErosionButton = QtWidgets.QPushButton(self.tab_12)
        self.GErosionButton.setObjectName("GErosionButton")
        self.gridLayout_9.addWidget(self.GErosionButton, 1, 0, 1, 1)
        self.GClosingButton = QtWidgets.QPushButton(self.tab_12)
        self.GClosingButton.setObjectName("GClosingButton")
        self.gridLayout_9.addWidget(self.GClosingButton, 3, 0, 1, 1)
        self.GReconstructionButton = QtWidgets.QPushButton(self.tab_12)
        self.GReconstructionButton.setObjectName("GReconstructionButton")
        self.gridLayout_9.addWidget(self.GReconstructionButton, 0, 1, 1, 1)
        self.GDilationButton = QtWidgets.QPushButton(self.tab_12)
        self.GDilationButton.setObjectName("GDilationButton")
        self.gridLayout_9.addWidget(self.GDilationButton, 0, 0, 1, 1)
        self.GOpeningButton = QtWidgets.QPushButton(self.tab_12)
        self.GOpeningButton.setObjectName("GOpeningButton")
        self.gridLayout_9.addWidget(self.GOpeningButton, 2, 0, 1, 1)
        self.MorphologicalGradientButton = QtWidgets.QPushButton(self.tab_12)
        self.MorphologicalGradientButton.setObjectName("MorphologicalGradientButton")
        self.gridLayout_9.addWidget(self.MorphologicalGradientButton, 1, 1, 1, 1)
        self.ConditionalDilationButton = QtWidgets.QPushButton(self.tab_12)
        self.ConditionalDilationButton.setObjectName("ConditionalDilationButton")
        self.gridLayout_9.addWidget(self.ConditionalDilationButton, 2, 1, 1, 1)
        self.gridLayout_6.addLayout(self.gridLayout_9, 1, 0, 1, 1)
        self.FunctionWidget.addTab(self.tab_12, "")
        self.gridLayout_2.addWidget(self.FunctionWidget, 0, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1354, 22))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menuImport_Image = QtWidgets.QMenu(self.menu)
        self.menuImport_Image.setObjectName("menuImport_Image")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionExport_Image = QtWidgets.QAction(MainWindow)
        self.actionExport_Image.setObjectName("actionExport_Image")
        self.actionImport_jpg = QtWidgets.QAction(MainWindow)
        self.actionImport_jpg.setObjectName("actionImport_jpg")
        self.actionImport_bmp = QtWidgets.QAction(MainWindow)
        self.actionImport_bmp.setObjectName("actionImport_bmp")
        self.menuImport_Image.addAction(self.actionImport_jpg)
        self.menuImport_Image.addAction(self.actionImport_bmp)
        self.menu.addAction(self.menuImport_Image.menuAction())
        self.menu.addAction(self.actionExport_Image)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        self.OriWidget.setCurrentIndex(0)
        self.ProWidget.setCurrentIndex(0)
        self.FunctionWidget.setCurrentIndex(3)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "BI23-Project"))
        self.OriImgName.setText(_translate("MainWindow", "原始图像"))
        self.ProImgName.setText(_translate("MainWindow", "处理后图像"))
        self.OriImg.setText(_translate("MainWindow", "请导入一张图片"))
        self.OriWidget.setTabText(self.OriWidget.indexOf(self.tab_3), _translate("MainWindow", "图像"))
        self.OriWidget.setTabText(self.OriWidget.indexOf(self.tab_4), _translate("MainWindow", "灰度直方图"))
        self.ProWidget.setTabText(self.ProWidget.indexOf(self.tab_7), _translate("MainWindow", "图像"))
        self.ProWidget.setTabText(self.ProWidget.indexOf(self.tab_8), _translate("MainWindow", "灰度直方图"))
        self.ConfirmButton.setText(_translate("MainWindow", "确认修改"))
        self.EntropyButton.setText(_translate("MainWindow", "Entropy Method"))
        self.OtsuButton.setText(_translate("MainWindow", "Otsu Method"))
        self.FunctionWidget.setTabText(self.FunctionWidget.indexOf(self.tab_9), _translate("MainWindow", "阈值处理"))
        self.SobelButton.setText(_translate("MainWindow", "Sobel Filter"))
        self.MedianButton.setText(_translate("MainWindow", "Median Filter"))
        self.FunctionWidget.setTabText(self.FunctionWidget.indexOf(self.tab_11), _translate("MainWindow", "图像滤波"))
        self.DilationButton.setText(_translate("MainWindow", "Dilation"))
        self.DistanceTransformButton.setText(_translate("MainWindow", "Distance Transform"))
        self.SkeletonRestorationButton.setText(_translate("MainWindow", "Skeleton restoration"))
        self.ErosionButton.setText(_translate("MainWindow", "Erosion"))
        self.OpeningButton.setText(_translate("MainWindow", "Opening"))
        self.ClosingButton.setText(_translate("MainWindow", "Closing"))
        self.SkeletonButton.setText(_translate("MainWindow", "Skeleton"))
        self.EdgeDetectionButton.setText(_translate("MainWindow", "Edge Detection"))
        self.FunctionWidget.setTabText(self.FunctionWidget.indexOf(self.tab_10), _translate("MainWindow", "形态学处理"))
        self.GErosionButton.setText(_translate("MainWindow", "Grayscale Erosion"))
        self.GClosingButton.setText(_translate("MainWindow", "Grayscale Closing"))
        self.GReconstructionButton.setText(_translate("MainWindow", "Grayscale Reconstruction"))
        self.GDilationButton.setText(_translate("MainWindow", "Grayscale Dilation"))
        self.GOpeningButton.setText(_translate("MainWindow", "Grayscale Opening"))
        self.MorphologicalGradientButton.setText(_translate("MainWindow", "Morphological Gradient"))
        self.ConditionalDilationButton.setText(_translate("MainWindow", "Conditional Dilation"))
        self.FunctionWidget.setTabText(self.FunctionWidget.indexOf(self.tab_12), _translate("MainWindow", "灰度级形态学"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.menuImport_Image.setTitle(_translate("MainWindow", "Import Image"))
        self.actionExport_Image.setText(_translate("MainWindow", "Export Image"))
        self.actionImport_jpg.setText(_translate("MainWindow", "Import .jpg"))
        self.actionImport_bmp.setText(_translate("MainWindow", "Import .bmp"))
