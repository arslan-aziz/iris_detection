import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.image as img
#import PIL.Image as pil
import cv2

#load image into ndarray and display
path='MMUIrisDatabase/MMU Iris Database/2/left/bryanl1.bmp'
im1=cv2.imread(path)
dims=im1.shape
print(type(im1),dims)
#plt.imshow(im1)
#plt.show()
#display color channels
fig = plt.figure()
#for i in range(1,4):
#    fig.add_subplot(1,3,i)
#    plt.imshow(im1[...,i-1])
#plt.show()

#convert to grayscale
grayArray=np.array([.299,.587,.114])
im1gray=np.dot(im1[...,:3],grayArray)
print(type(im1gray),im1gray.shape)
#plt.imshow(im1gray)
#plt.show()
print(im1gray.max(),im1gray.min())
#plt.hist(im1gray.ravel())
#plt.show()

#find pupil
#divide image into smaller regions
numGrid=20
x=np.floor(np.linspace(0,dims[0],numGrid+1)).astype(int)
y=np.floor(np.linspace(0,dims[1],numGrid+1)).astype(int)
xv,yv=np.meshgrid(x,y)

gridAvg=np.zeros((numGrid,numGrid))
#calculate mean in each region
for a in range(numGrid):
    for b in range(numGrid):
        gridAvg[a,b] = np.mean(im1gray[xv[a,b]:xv[a,b+1], yv[a,b]:yv[a+1,b]])

#for idx,i in enumerate(np.nditer(gridAvg,order='C',op_flags=['readwrite']),0):
#    i=np.mean(im1gray[xv[idx]:xv[idx+1],yv[idx]:yv[idx+21]])
print(gridAvg.shape)

#find image subscript of minimum avg. intensity
low=np.unravel_index(gridAvg.argmin(),(numGrid,numGrid))
a=low[0]
b=low[1]
sd=np.std(im1gray[xv[a,b]:xv[a,b+1], yv[a,b]:yv[a+1,b]])
x_cen_temp=xv[low[0],low[1]]
y_cen_temp=yv[low[0],low[1]]
print(x_cen_temp,y_cen_temp)
#create binary iris
#TODO: use connected components and region thresholding to select ONLY iris (largest component)
_,mask=cv2.threshold(im1gray,gridAvg.min()+1.96*sd,1,cv2.THRESH_BINARY)
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
iris_mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
iris_mask=cv2.morphologyEx(iris_mask,cv2.MORPH_CLOSE,kernel)

#visualize
fig.add_subplot(1,4,1)
plt.imshow(gridAvg.T)
plt.annotate('>',low)
fig.add_subplot(1,4,2)
plt.imshow(im1gray)
#plt.annotate('>',(y_cen_temp,x_cen_temp))
plt.annotate('local max', xy=(y_cen_temp, x_cen_temp), xytext=(3, 1.5),
            arrowprops=dict(facecolor='black', shrink=0.05))
fig.add_subplot(1,4,3)
plt.imshow(mask)
fig.add_subplot(1,4,4)
plt.imshow(iris_mask)
plt.show()


#find pupil shape
#step 1
diff=np.array([1,-1])
#row_temp=iris_mask[y_cen_temp-10:y_cen_temp+10,:]
#row_temp=np.average(row_temp,0)
row_temp=iris_mask[x_cen_temp,:]
row_temp_diff=np.absolute(np.convolve(diff,row_temp,'valid'))
row_temp_max=np.argwhere(row_temp_diff==1)
left_temp=row_temp_max[0]
right_temp=row_temp_max[-1]
print(left_temp,right_temp)
fig.add_subplot(1,3,1)
plt.plot(np.arange(0,row_temp_diff.size),row_temp_diff)
#step 2
y_cen=int((left_temp+right_temp)/2)
col_temp=iris_mask[:,y_cen]
col_temp_diff=np.absolute(np.convolve(diff,col_temp,'valid'))
col_temp_max=np.argwhere(col_temp_diff==1)
top_temp=col_temp_max[0]
bot_temp=col_temp_max[-1]
print(top_temp,bot_temp)
fig.add_subplot(1,3,2)
plt.plot(np.arange(0,col_temp_diff.size),col_temp_diff)
#step 3
x_cen=int((top_temp+bot_temp)/2)
row_temp=iris_mask[x_cen,:]
row_temp_diff=np.absolute(np.convolve(diff,row_temp,'valid'))
row_temp_max=np.argwhere(row_temp_diff==1)
left_temp=row_temp_max[0]
right_temp=row_temp_max[-1]
print(left_temp,right_temp)
fig.add_subplot(1,3,3)
plt.plot(np.arange(0,row_temp_diff.size),row_temp_diff)
plt.show()

#step 4
r=((right_temp-left_temp)+(bot_temp-top_temp))/4
print(x_cen,y_cen,r)
plt.imshow(im1gray)
fig=plt.gcf()
ax=fig.gca()
circle=plt.Circle((y_cen,x_cen),r,color='blue',fill=False)
ax.add_artist(circle)
plt.show()

#segmentation of iris from sclera
#median filter image
im1_med=np.float64(cv2.medianBlur(cv2.convertScaleAbs(im1gray),3))
#draw concentric circles
#r_test=np.arange(r+1,120,2)
#theta_test=np.arange(-1*np.pi/4,0,0.05)
#circle_polar=lambda theta,r: np.array([r*np.cos(theta),r*np.sin(theta)])
#print(circle_polar(np.pi/4,1))

#check difference across relevant sectors on left and right

#generate circle points
r=10
x=r
y=0
xPoints=[]
yPoints=[]
while x>=y:
    xPoints.append(x)
    yPoints.append(y)
    radius_error=x^2+y^2-r^2
    if 2*(radius_error+(2*y+1))+(1-2*x)>0:
        y+=1
        x-=1
    else:
        y+=1




#take laplacian of gaussian of eye
#im1_gauss=cv2.GaussianBlur(im1gray,(3,3),1)
#im1_log=cv2.Laplacian(cv2.convertScaleAbs(im1_gauss),cv2.CV_8U,3)
#fig.add_subplot(1,3,1)
#plt.imshow(im1_log)
#threshold zero crossings
#_,im1_edge=cv2.threshold(im1_log,2,1,cv2.THRESH_BINARY)
#fig.add_subplot(1,3,2)
#plt.imshow(im1_edge)
#_,b=cv2.connectedComponents(im1_edge,4)
#fig.add_subplot(1,3,3)
#plt.imshow(b)
#plt.show()






