from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5 import Qt
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget,QTableWidget,QTableWidgetItem,QGraphicsScene,QGraphicsPixmapItem,QFileDialog
from Design import Ui_MainWindow

import os
import io as _io
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from xlrd import open_workbook
from openpyxl.reader.excel import load_workbook

from skimage import data, img_as_float,io
from skimage.measure import compare_ssim as SSIM2

from sklearn.metrics import mean_squared_error as MSE
from PIL import Image
import scipy
import pandas as pd
from scipy import ndimage
from sklearn import decomposition
from skimage import data
from skimage import color
from skimage.segmentation import clear_border
from skimage.morphology import label, closing, square
from skimage.measure import regionprops
import random
import matplotlib.image as IMG
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
from skimage.feature import greycomatrix, greycoprops
from skimage import data
from PIL.ImageQt import ImageQt
from skimage.feature import daisy
import pickle
from sklearn.metrics import jaccard_similarity_score
from skimage.feature import daisy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class override_graphicsScene (Qt.QGraphicsScene):
    def __init__(self,parent = None):
        super(override_graphicsScene,self).__init__(parent)

    def mousePressEvent(self, event):
        super(override_graphicsScene, self).mousePressEvent(event)
        print(event.pos())

class MainWindow(QWidget,Ui_MainWindow):
    
    temp_path = "image.png"
    temp_path_2 = "image_2.png"
    temp_path_3 = "image_3.png"
    
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)


        
        self.btn_features_apply.clicked.connect(self.button_features_apply)
        self.btn_features_apply_2.clicked.connect(self.button_features_apply_2)


    
    
        
    def onSelected(self,row,column):
        image_name = self.table_keypoint_result.item(row,0).text()
        scene = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img_keypoint_result.size())
        self.img_keypoint_result.setScene(scene)
        
        jaccard_score = self.table_keypoint_result.item(row,5).text()
        self.lbl_keypoint_jaccard.setText(jaccard_score)
        
    

        
    #FEATURES
    directory = "./objects/features/"
    path = "./objects/features/"

    def button_features_apply_2(self):
        self.table_features.clear()
        self.table_features_labels.clear()
        self.table_features_y.clear()       
        features = []
        features += [0 for i in range(0,100)]       
        features_label = []        
        list_x,list_y=[],[]
        x,y=[],[]
        j = 0        
        count_features = 0        
        folder_list = os.listdir(self.directory)
        for i,folder in enumerate(folder_list):
            files_list = os.listdir(self.path+folder)
            features_label.append([folder,i])           
            for j,file in enumerate(files_list):
                img = cv2.imread(self.directory+folder+"/"+file, cv2.COLOR_BGR2GRAY)
                desc = greycomatrix(img, [2], [0], 256, symmetric=True, normed=True)
                desc = desc.flatten()
                list_x.append([desc[0],desc[len(desc)-1]])
                d_temp = np.array(desc)                
                list_y.append(file)
                y.append(int(i))
                features[j]=d_temp
                j+=1                               
                count_features += 1
                self.lbl_features_count.setText(str(count_features))        
        features = np.array(features)  
        print("x boyut:",str(len(list_x)))        
        self.table_features_2.setColumnCount(len(list_x[0])) #len(features[0])
        self.table_features_2.setRowCount(len(list_x))
        for _i,row in enumerate(list_x):
            for _j,value in enumerate(row):
                self.table_features_2.setItem(_i,_j, QTableWidgetItem(str(value)))               #break
        self.table_features_2.horizontalHeader().setStretchLastSection(True)
        self.table_features_2.resizeColumnsToContents()
        self.table_features_2.setHorizontalHeaderLabels(['feature_1','feature_2'])
        
        #print(y)
        self.table_features_y_2.setColumnCount(1)
        self.table_features_y_2.setRowCount(len(y))
        for _i,row in enumerate(y):
            self.table_features_y_2.setItem(_i,0, QTableWidgetItem(str(row)))
        self.table_features_y_2.horizontalHeader().setStretchLastSection(True)
        self.table_features_y_2.resizeColumnsToContents()
        self.table_features_y_2.setHorizontalHeaderLabels(['y'])
        
        #print(features_label)
        self.table_features_labels_2.setColumnCount(len(features_label[0]))
        self.table_features_labels_2.setRowCount(len(features_label))
        for _i,row in enumerate(features_label):
            for _j, value in enumerate(row):
                self.table_features_labels_2.setItem(_i,_j, QTableWidgetItem(str(value)))
        self.table_features_labels_2.horizontalHeader().setStretchLastSection(True)
        self.table_features_labels_2.resizeColumnsToContents()
        self.table_features_labels_2.setHorizontalHeaderLabels(['Label','Key'])
        
        X_train, X_test, y_train, y_test = train_test_split(list_x, y, test_size = 1/3, random_state = 0)
            
        accurity = self.ALGORITHM_RANDOM_FOREST(X_train, X_test, y_train, y_test)
        print("Başarı oranı:",str(accurity))
    def button_features_apply(self):
        self.table_features.clear()
        self.table_features_labels.clear()
        self.table_features_y.clear()        
        features = []
        features += [0 for i in range(0,100)]        
        features_label = []
       
        list_x,list_y=[],[]
        x,y=[],[]
        j = 0        
        count_features = 0       
        folder_list = os.listdir(self.directory)
        for i,folder in enumerate(folder_list):
            files_list = os.listdir(self.path+folder)          
            features_label.append([folder,i])            
            for j,file in enumerate(files_list):
                img = cv2.imread(self.directory+folder+"/"+file, cv2.COLOR_BGR2GRAY)
               
                desc = self.gabor_features(img,self.build_filters())
                d = decomposition.PCA(2).fit_transform(desc)                    #print(str(j),"----------------\n",d)
                list_x.append([d[0][0],d[0][1]])
                d_temp = np.array([d[0][0],d[0][1]]) 
                list_y.append(file)
                y.append(int(i))
                features[j]=d_temp
                j+=1
                               
                count_features += 1
                self.lbl_features_count.setText(str(count_features))
        
        features = np.array(features)      
        print("x boyut:",str(len(list_x)))
        
        self.table_features.setColumnCount(len(list_x[0])) #len(features[0])
        self.table_features.setRowCount(len(list_x))

        for _i,row in enumerate(list_x):
            for _j,value in enumerate(row):
                self.table_features.setItem(_i,_j, QTableWidgetItem(str(value)))
                #break
        self.table_features.horizontalHeader().setStretchLastSection(True)
        self.table_features.resizeColumnsToContents()
        self.table_features.setHorizontalHeaderLabels(['feature_1','feature_2'])
        
        #print(y)
        self.table_features_y.setColumnCount(1)
        self.table_features_y.setRowCount(len(y))
        for _i,row in enumerate(y):
            self.table_features_y.setItem(_i,0, QTableWidgetItem(str(row)))
        self.table_features_y.horizontalHeader().setStretchLastSection(True)
        self.table_features_y.resizeColumnsToContents()
        self.table_features_y.setHorizontalHeaderLabels(['y'])
        
        #print(features_label)
        self.table_features_labels.setColumnCount(len(features_label[0]))
        self.table_features_labels.setRowCount(len(features_label))
        for _i,row in enumerate(features_label):
            for _j, value in enumerate(row):
                self.table_features_labels.setItem(_i,_j, QTableWidgetItem(str(value)))
        self.table_features_labels.horizontalHeader().setStretchLastSection(True)
        self.table_features_labels.resizeColumnsToContents()
        self.table_features_labels.setHorizontalHeaderLabels(['Label','Key'])
        
        X_train, X_test, y_train, y_test = train_test_split(list_x, y, test_size = 1/3, random_state = 0)
            
        accurity = self.ALGORITHM_RANDOM_FOREST(X_train, X_test, y_train, y_test)
        print("Başarı oranı:",str(accurity))


    def ALGORITHM_RANDOM_FOREST(self,_x_train,_x_test,_y_train,_y_test):
        model = RandomForestClassifier()
        model.fit(_x_train,_y_train)
        y_pred = model.predict(_x_test)
        accurity = accuracy_score(_y_test,y_pred)
        return str(round(accurity*100,2))

    def build_filters(self):
        filters = []
        ksize = 31
        for theta in np.arange(0, np.pi, np.pi / 16):
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
        return filters
         
    def gabor_features(self,img, filters):
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
        return accum


        
    
    def step_II(self,img,path):
        teval,img = cv2.threshold(img,10,255, cv2.THRESH_BINARY)
        if(path != None):
            cv2.imwrite(path,img)
        return img
    
    def step_I(self,img,path):
        #img = color.rgb2gray(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        img = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img[0])
        if(path != None):
            cv2.imwrite(path,img)
        return img
    


    
    def key_show(self,img,key):
        img = cv2.drawKeypoints(img, key, None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    #metodlar
    def show_image_path(self,img_path,size):
        self.pixmap = Qt.QPixmap()
        self.pixmap.load(img_path)
        self.pixmap = self.pixmap.scaled(size, Qt.Qt.KeepAspectRatioByExpanding,transformMode=QtCore.Qt.SmoothTransformation)
        self.graphicsPixmapItem = Qt.QGraphicsPixmapItem(self.pixmap)
        self.graphicsScene = override_graphicsScene(self)
        self.graphicsScene.addItem(self.graphicsPixmapItem)
        return self.graphicsScene
    
    def ssim(self,img1,img2):
        img_1 = np.asarray(img1)#cv2.imread(img1)
        img_2 = np.asarray(img2)#cv2.imread(img2)

        img_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
        img_2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)
        try:
            if(not img_1 is None and not img_2 is None):
                if(img_1.size == img_2.size):
                    return round(SSIM2(img_1,img_2),2)
            else:
                return 0.0
        except ValueError:
            print("Invalid Entry - try again")
            return 0.0
        return 0.0
    

    
    def jaccard(self,img1,img2):
        img_true=np.array(img1).ravel()
        img_pred=np.array(img2).ravel()
        iou = jaccard_similarity_score(img_true, img_pred)
        return iou
    
