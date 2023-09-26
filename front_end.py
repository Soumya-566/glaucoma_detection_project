from keras.models import load_model
# if cnn model is used we load cnn model otherwise rnn
model=load_model('C:\\glaucoma_project\\cnn_model.h5')
import pandas as pd;
import scipy.signal
c1=""
d1=""
file_name=""
def segment1(image,plot_seg,plot_hist,name):

    image = image[400:1500,400:1500,:] 

    Abo,Ago,Aro = cv2.split(image) 
    M = 100
    filter = scipy.signal.gaussian(M, std=6)
    STDf = filter.std()
    

    Ar = Aro - Aro.mean() - Aro.std() 
    Mr = Ar.mean()          
    SDr = Ar.std()                          
    Thr = 0.5*M - STDf - SDr            

    
    M = 30
    filter_cup = scipy.signal.gaussian(M, std=6) 
    STDf = filter_cup.std()  

    Ag = Ago - Ago.mean() - Ago.std()
    Mg = Ag.mean()                        
    SDg = Ag.std()                        
    Thg = 0.5*M +2*STDf + 2*SDg + Mg     
    #print(Thg)
    
    
    hist,bins = np.histogram(Ag.ravel(),256,[0,256])  
    histr,binsr = np.histogram(Ar.ravel(),256,[0,256])


    smooth_hist_g=np.convolve(filter,hist)  
    smooth_hist_r=np.convolve(filter_cup,histr) 
    
    
    r,c = Ag.shape
    Dd = np.zeros(shape=(r,c)) 
    Dc = np.zeros(shape=(r,c)) 

    for i in range(1,r):
        for j in range(1,c):
            if Ar[i,j]>Thr:
                Dd[i,j]=255
            else:
                Dd[i,j]=0

    for i in range(1,r):
        for j in range(1,c):
        
            if Ag[i,j]>Thg:
                Dc[i,j]=255
            else:
                Dc[i,j]=0
  
    from datetime import datetime
    x = datetime.now()
    #global c1,d1
    c= 'G:\\c1'+x.strftime("%Y_%m_%d%I%M%S_%p")+'.png'
    d= 'G:\\d'+x.strftime("%Y_%m_%d%I%M%S_%p")+'.png'
    #f = open(r)
    global c1,d1
    c1=c
    d1=d
    with open(c1, 'w') as fp:
        pass
    fp.close()
    with open(d1, 'w') as fp:
        pass
    fp.close()

    cv2.imwrite(d1,Dd)
    plt.imsave(c1,Dc)
    
    
def cdr(cup,disc,plot):
    
    #morphological closing and opening operations
    R1 = cv2.morphologyEx(cup, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)), iterations = 1)
    r1 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations = 1)
    R2 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,21)), iterations = 1)
    r2 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,1)), iterations = 1)
    R3 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(33,33)), iterations = 2)
    img = R3
    ret,thresh = cv2.threshold(img,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #Getting all possible contours in the segmented image
    cup_diameter = 0
    largest_area = 0
    el_cup = contours[0]
    if len(contours) != 0:
        for i in range(len(contours)):
            if len(contours[i]) >= 5:
                area = cv2.contourArea(contours[i]) 
                if (area>largest_area):
                    largest_area=area
                    index = i
                    el_cup = cv2.fitEllipse(contours[i])
                
    cv2.ellipse(img,el_cup,(140,60,150),3)  
    x,y,w,h = cv2.boundingRect(contours[index]) 
    cup_diameter = max(w,h)
    cubx=w
    cuby=h

    #morphological closing and opening operations
    R1 = cv2.morphologyEx(disc, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)), iterations = 1)
    r1 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations = 1)
    R2 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,21)), iterations = 1)
    r2 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,1)), iterations = 1)
    R3 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(33,33)), iterations = 1)
    r3 = cv2.morphologyEx(R3, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(43,43)), iterations = 1)
    img2 = r3
    
    ret,thresh = cv2.threshold(img2,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    disk_diameter = 0
    largest_area = 0
    el_disc = el_cup
    if len(contours) != 0:
          for i in range(len(contours)):
            if len(contours[i]) >= 5:
                area = cv2.contourArea(contours[i]) 
                if (area>largest_area):
                    largest_area=area
                    index = i
                    el_disc = cv2.fitEllipse(contours[i])
                    
            cv2.ellipse(img2,el_disc,(140,60,150),10) 
            x,y,w,h = cv2.boundingRect(contours[index]) 
    disk_diameter = max(w,h)
    diskx=w
    disky=h
                

        
    if(disk_diameter == 0): return 1 
    cdr = cup_diameter/disk_diameter 
    return cdr,cubx,cuby,diskx,disky
import numpy as np
from skimage.feature import greycomatrix
import cv2 as cv
def entropy(file_names): 
    img = cv.imread(file_names,0)
    glcm = np.squeeze(greycomatrix(img, distances=[1], 
                               angles=[0], symmetric=True, 
                               normed=True))
    entropy = -np.sum(glcm*np.log2(glcm + (glcm==0)))
    return entropy
def DDE(image):
    img = cv.imread(image,0)
    sobelx8u = cv.Sobel(img,cv.CV_8U,1,0,ksize=5)
    sobelx64f = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)
    return sobel_8u
from scipy.linalg import norm
from scipy import sum, average
def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng
def compare_images(img1, img2):
    
    img1 = normalize(img1)
    img2 = normalize(img2)
    
    diff = img1 - img2  
    m_norm = sum(abs(diff))
    z_norm = norm(diff.ravel(), 0)  
    return (m_norm, z_norm)
import streamlit as st
import numpy as np
import cv2
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("A Novel Optic Disc and Optic Cup Segmentation Technique to Diagnose Glaucoma Using Deep Learning Algorithms:")
file = st.file_uploader("Please upload an image(png) file", type=["png"])
st.write(file)
#st.image(
#option = st.selectbox(
#     'Choose an alogrithm',
#     ('RNN',))
#st.write('You selected:', option)
if file is None:
    st.text("You haven't uploaded a png image file")
else:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    o = cv2.imdecode(file_bytes, 1)
    from datetime import datetime
    #image = file.read()
    st.image(o)


    x = datetime.now()
    import os
    #global file_name
    file_name1= 'G:\\img'+x.strftime("%Y_%m_%d%I%M%S_%p")+'.png'
    #f = open(r)
    #global file_name
    file_name=file_name1
    with open(file_name, 'w') as fp:
        pass
    fp.close()
    cv2.imwrite(file_name,o)
    #cv2.imshow('done',cv2.imread(file_name))

    #cv2.imwrite('C:\\Users\\kodandarao\\2img.png',o)
    #image = file.read()
    #img = st.image(image, caption='Sunrise by the mountains', use_column_width=True)
    #st.write(type(img))
    #imageI = Image.open(file)
    #prediction = import_and_predict(imageI, model)
    #pred = prediction[0][0]
    #import cv2
    #image = cv2.imread(o,1)
    #segment(o,True,True,"hai")
    #cup = cv2.imread('cup.png',0)
    #disc = cv2.imread('disk.png',0)
    #a=cdr(cup,disc,None)
    ##if(a[0]>0.5):
    #    st.write("You are under risk of glaucoma")
    #else:
    #    st.write("Healthy eye!!!")
    img=cv2.imread(file_name,1)
    import cv2
    import numpy as np
    import sys
    import os
    import glob
    import matplotlib.pyplot as plt
    import pandas as pd
    segment1(img,True,True,None)
    #global c1,d1
    cup = cv2.imread(c1,0) 
    disc = cv2.imread(d1,0)
    #st.write(c1)
    #st.write(d1)
    cv2.imshow('jj',cup)
    cv2.imshow('dd',disc)
    cdr_cal,cubx,cuby,diskx,disky = cdr(cup,disc,True)
    CDR = [] 
    VAL = [] 
    CDR1 = [] 
    CDR2 = [] 
    CDR3 = [] 
    CDR4 = [] 
    CDR.append(cdr_cal)
    CDR1.append(cubx)
    CDR2.append(cuby)
    CDR3.append(diskx)
    CDR4.append(disky)
    en=[]
    c=[]
    c1=[]
    c2=[]
    c3=[]
    #global file_name
    set_path =file_name
    entropys=entropy(set_path)
    en.append(entropys)
    img = cv2.imread(set_path,0)
    img = cv2.GaussianBlur(img,(3,3),0)
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  
    s,s1=compare_images(img,DDE(set_path))
    d,d1=compare_images(img,sobelx)
    c.append(s)
    c1.append(s1)
    c2.append(d)
    c3.append(d1)
    VAL=[]
    if CDR[0]>0.6:
        VAL.append(1)
    else:
        VAL.append(0)

    df=pd.DataFrame();
    #st.write(len(df))
    
    df["1"]=CDR1
    df["2"]=CDR2
    df["3"]=CDR3
    df["4"]=CDR4
    df["5"]=en
    df["6"]=c
    df["7"]=c1
    df["8"]=c2
    df["9"]=c3
    df["10"]=CDR
    df["tar"]=VAL
    #st.write(df)
    #("cdr=",CDR)
    train_set = df.tail(n=1)
    test_set = train_set
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0,1))
    test_set_scaled = sc.fit_transform(test_set)
    X_test = []
    y_test = []
    for i in range(len(test_set_scaled)):
        X_test.append(test_set_scaled[i][0:10])
        y_test.append(test_set_scaled[i][10])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    oo = model.predict(X_test)
    #print(X_test)
    #st.image(o, caption='Uploaded image')

    #st.write(o)
    if(CDR[-1]>0.6):
            st.markdown("**Prediction**: ***You are affected by glaucoma. Please consult an opthalmologist as soon as possible.***")
        
    else:
            st.markdown("**Prediction**:Your eye is Healthy. Great!!!")
    #st.write(oo[-1][-1])
    
