import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio.v2 as imageio
from itertools import islice
from skimage import feature

def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))
def lbp_features(img, radius=1, sampling_pixels=8):
    
    # LBP operates in single channel images so if RGB images are provided
    # we have to convert it to grayscale
    if (len(img.shape) > 2):
        img = img.astype(float)
        # RGB to grayscale convertion using Luminance
        img = img[:,:,0]*0.3 + img[:,:,1]*0.59 + img[:,:,2]*0.11

    # converting to uint8 type for 256 graylevels
    img = img.astype(np.uint8)
    
    # normalize values can also help improving description
    i_min = np.min(img)
    i_max = np.max(img)
    if (i_max - i_min != 0):
        img = (img - i_min)/(i_max-i_min)
    
    # compute LBP
    lbp = feature.local_binary_pattern(img, sampling_pixels, radius, method="uniform")
    
    # LBP returns a matrix with the codes, so we compute the histogram
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, sampling_pixels + 3), range=(0, sampling_pixels + 2))

    # normalization
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    # return the histogram of Local Binary Patterns
    return hist

def Euclidean_distance(p, q):
    dist = np.sqrt(np.sum(np.square(p-q)))
    return dist
def inter_distance(hist,hist1):
 m, sum, d = 0, 0, 0
 i = 0
 while i<len(hist) and i<len(hist1):
  m+=min(hist[i],hist1[i])
  sum+=hist[i]
  i+= 1
 d = 1 -( m / sum)

 return d


# lecture des images
image0 = imageio.imread('static/dataBase/image0.jpg')
image1 = imageio.imread('static/dataBase/image1.jpg')
image2 = imageio.imread('static/dataBase/image2.jpg')
image3 = imageio.imread('static/dataBase/image3.jpg')
image4 = imageio.imread('static/dataBase/image4.jpg')
image5 = imageio.imread('static/dataBase/image5.jpg')
image6 = imageio.imread('static/dataBase/image6.jpg')
image7 = imageio.imread('static/dataBase/image7.jpg')
image8 = imageio.imread('static/dataBase/image8.jpg')
image9 = imageio.imread('static/dataBase/image9.jpg')
image10 = imageio.imread('static/dataBase/image10.jpg')
'''
image11 = imageio.imread('static/dataBase/image11.jpg')
image12 = imageio.imread('static/dataBase/image12.jpg')
image13 = imageio.imread('static/dataBase/image13.jpg')
image14 = imageio.imread('static/dataBase/image14.jpg')
image15 = imageio.imread('static/dataBase/image15.jpg')
image16 = imageio.imread('static/dataBase/image16.jpg')
image17 = imageio.imread('static/dataBase/image17.jpg')
image18 = imageio.imread('static/dataBase/image18.jpg')
image19 = imageio.imread('static/dataBase/image19.jpg')
image20= imageio.imread('static/dataBase/image20.jpg')
image21 = imageio.imread('static/dataBase/image21.jpg')
image22 = imageio.imread('static/dataBase/image22.jpg')
image23 = imageio.imread('static/dataBase/image23.jpg')
image24 = imageio.imread('static/dataBase/image24.jpg')
image25 = imageio.imread('static/dataBase/image25.jpg')
image25 = imageio.imread('static/dataBase/image25.jpg')
image26 = imageio.imread('static/dataBase/image26.jpg')
image27 = imageio.imread('static/dataBase/image27.jpg')
image28 = imageio.imread('static/dataBase/image28.jpg')
image29 = imageio.imread('static/dataBase/image29.jpg')
image30 = imageio.imread('static/dataBase/image30.jpg')
image31 = imageio.imread('static/dataBase/image31.jpg')
image32 = imageio.imread('static/dataBase/image32.jpg')
image33 = imageio.imread('static/dataBase/image33.jpg')
image34 = imageio.imread('static/dataBase/image34.jpg')
image35 = imageio.imread('static/dataBase/image35.jpg')
image36 = imageio.imread('static/dataBase/image36.jpg')
image37 = imageio.imread('static/dataBase/image37.jpg')
image38 = imageio.imread('static/dataBase/image38.jpg')
image39 = imageio.imread('static/dataBase/image39.jpg')
image40 = imageio.imread('static/dataBase/image40.jpg')
image41 = imageio.imread('static/dataBase/image41.jpg')
image42 = imageio.imread('static/dataBase/image42.jpg')
image43 = imageio.imread('static/dataBase/image43.jpg')
image44 = imageio.imread('static/dataBase/image44.jpg')
image45 = imageio.imread('static/dataBase/image45.jpg')
image46 = imageio.imread('static/dataBase/image46.jpg')
image47 = imageio.imread('static/dataBase/image47.jpg')
image48 = imageio.imread('static/dataBase/image48.jpg')
image49 = imageio.imread('static/dataBase/image49.jpg')
image50 = imageio.imread('static/dataBase/image50.jpg')
image51 = imageio.imread('static/dataBase/image51.jpg')
image52 = imageio.imread('static/dataBase/image52.jpg')
image53 = imageio.imread('static/dataBase/image53.jpg')
image54 = imageio.imread('static/dataBase/image54.jpg')
image55 = imageio.imread('static/dataBase/image55.jpg')
image56 = imageio.imread('static/dataBase/image56.jpg')
image57 = imageio.imread('static/dataBase/image57.jpg')
image58 = imageio.imread('static/dataBase/image58.jpg')
image59 = imageio.imread('static/dataBase/image59.jpg')
image60 = imageio.imread('static/dataBase/image60.jpg')
image61 = imageio.imread('static/dataBase/image61.jpg')
image62 = imageio.imread('static/dataBase/image62.jpg')
image63 = imageio.imread('static/dataBase/image63.jpg')
image64 = imageio.imread('static/dataBase/image64.png')
image65 = imageio.imread('static/dataBase/image65.jpg')
image66 = imageio.imread('static/dataBase/image66.jpg')
image67 = imageio.imread('static/dataBase/image67.jpg')
image68 = imageio.imread('static/dataBase/image68.jpg')
image69 = imageio.imread('static/dataBase/image69.jpg')
image70 = imageio.imread('static/dataBase/image70.jpg')
image71 = imageio.imread('static/dataBase/image71.jpg')
image72 = imageio.imread('static/dataBase/image72.jpg')
image73 = imageio.imread('static/dataBase/image73.jpg')
image74 = imageio.imread('static/dataBase/image74.jpg')
image75 = imageio.imread('static/dataBase/image75.jpg')
image76 = imageio.imread('static/dataBase/image76.jpg')
image77 = imageio.imread('static/dataBase/image77.jpg')
image78 = imageio.imread('static/dataBase/image78.jpg')
image79 = imageio.imread('static/dataBase/image79.jpg')
image80 = imageio.imread('static/dataBase/image80.jpg')
image81 = imageio.imread('static/dataBase/image81.jpg')
image82 = imageio.imread('static/dataBase/image82.jpg')
image83 = imageio.imread('static/dataBase/image83.jpg')
image84 = imageio.imread('static/dataBase/image84.jpg')
image85 = imageio.imread('static/dataBase/image85.jpg')
image86 = imageio.imread('static/dataBase/image86.jpg')
image87 = imageio.imread('static/dataBase/image87.jpg')
image88 = imageio.imread('static/dataBase/image88.jpg')
image89 = imageio.imread('static/dataBase/image89.jpg')
image90 = imageio.imread('static/dataBase/image90.jpg')
image91 = imageio.imread('static/dataBase/image91.jpg')
image92 = imageio.imread('static/dataBase/image92.jpg')
image93 = imageio.imread('static/dataBase/image93.jpg')
image94 = imageio.imread('static/dataBase/image94.jpg')
image95 = imageio.imread('static/dataBase/image95.jpg')
image96 = imageio.imread('static/dataBase/image96.jpg')
image97 = imageio.imread('static/dataBase/image97.jpg')
image98 = imageio.imread('static/dataBase/image98.jpg')
image99 = imageio.imread('static/dataBase/image99.jpg')
'''
#calcul des histogrammes 
histogram0 = lbp_features(image0,2,8)
histogram1 = lbp_features(image1,2,8)
histogram2 = lbp_features(image2,2,8)
histogram3 = lbp_features(image3,2,8)
histogram4 = lbp_features(image4,2,8)
histogram5 = lbp_features(image5,2,8)
histogram6 = lbp_features(image6,2,8)
histogram7 = lbp_features(image7,2,8)
histogram8 = lbp_features(image8,2,8)
histogram9 = lbp_features(image9,2,8)
histogram10 = lbp_features(image10,2,8)
'''
histogram11 = lbp_features(image11,2,8)
histogram12 = lbp_features(image12,2,8)
histogram13 = lbp_features(image13,2,8)
histogram14 = lbp_features(image14,2,8)
histogram15 = lbp_features(image15,2,8)
histogram16 = lbp_features(image16,2,8)
histogram17 = lbp_features(image17,2,8)
histogram18 = lbp_features(image18,2,8)
histogram19 = lbp_features(image19,2,8)
histogram20 = lbp_features(image20,2,8)
histogram21 = lbp_features(image21,2,8)
histogram22 = lbp_features(image22,2,8)
histogram23 = lbp_features(image23,2,8)
histogram24 = lbp_features(image24,2,8)
histogram25 = lbp_features(image25,2,8)
histogram26 = lbp_features(image26,2,8)
histogram27 = lbp_features(image27,2,8)
histogram28 = lbp_features(image28,2,8)
histogram29 = lbp_features(image29,2,8)
histogram30 = lbp_features(image30,2,8)
histogram31 = lbp_features(image31,2,8)
histogram32 = lbp_features(image32,2,8)
histogram33 = lbp_features(image33,2,8)
histogram34 = lbp_features(image34,2,8)
histogram35 = lbp_features(image35,2,8)
histogram36 = lbp_features(image36,2,8)
histogram37 = lbp_features(image37,2,8)
histogram38 = lbp_features(image38,2,8)
histogram39 = lbp_features(image39,2,8)
histogram40 = lbp_features(image40,2,8)
histogram41 = lbp_features(image41,2,8)
histogram42 = lbp_features(image42,2,8)
histogram43 = lbp_features(image43,2,8)
histogram44 = lbp_features(image44,2,8)
histogram45 = lbp_features(image45,2,8)
histogram46 = lbp_features(image46,2,8)
histogram47 = lbp_features(image47,2,8)
histogram48 = lbp_features(image48,2,8)
histogram49 = lbp_features(image49,2,8)
histogram50 = lbp_features(image50,2,8)
histogram51 = lbp_features(image51,2,8)
histogram52 = lbp_features(image52,2,8)
histogram53 = lbp_features(image53,2,8)
histogram54 = lbp_features(image54,2,8)
histogram55 = lbp_features(image55,2,8)
histogram56 = lbp_features(image56,2,8)
histogram57 = lbp_features(image57,2,8)
histogram58 = lbp_features(image58,2,8)
histogram59 = lbp_features(image59,2,8)
histogram60 = lbp_features(image60,2,8)
histogram61 = lbp_features(image61,2,8)
histogram62 = lbp_features(image62,2,8)
histogram63 = lbp_features(image63,2,8)
histogram64 = lbp_features(image64,2,8)
histogram65 = lbp_features(image65,2,8)
histogram66 = lbp_features(image66,2,8)
histogram67 = lbp_features(image67,2,8)
histogram68 = lbp_features(image68,2,8)
histogram69 = lbp_features(image69,2,8)
histogram70 = lbp_features(image70,2,8)
histogram71 = lbp_features(image71,2,8)
histogram72 = lbp_features(image72,2,8)
histogram73 = lbp_features(image73,2,8)
histogram74 = lbp_features(image74,2,8)
histogram75 = lbp_features(image75,2,8)
histogram76 = lbp_features(image76,2,8)
histogram77 = lbp_features(image77,2,8)
histogram78 = lbp_features(image78,2,8)
histogram79 = lbp_features(image79,2,8)
histogram80 = lbp_features(image80,2,8)
histogram81 = lbp_features(image81,2,8)
histogram82 = lbp_features(image82,2,8)
histogram83 = lbp_features(image83,2,8)
histogram84 = lbp_features(image84,2,8)
histogram85 = lbp_features(image85,2,8)
histogram86 = lbp_features(image86,2,8)
histogram87 = lbp_features(image87,2,8)
histogram88 = lbp_features(image88,2,8)
histogram89 = lbp_features(image89,2,8)
histogram90 = lbp_features(image90,2,8)
histogram91 = lbp_features(image91,2,8)
histogram92 = lbp_features(image92,2,8)
histogram93 = lbp_features(image93,2,8)
histogram94 = lbp_features(image94,2,8)
histogram95 = lbp_features(image95,2,8)
histogram96 = lbp_features(image96,2,8)
histogram97 = lbp_features(image97,2,8)
histogram98 = lbp_features(image98,2,8)
histogram99 = lbp_features(image99,2,8)
'''
thisdicttexture = {
  "image0": histogram0,
  "image1": histogram1,
  "image2": histogram2,
  "image3": histogram3,
  "image4": histogram4,
  "image5": histogram5,
  "image6": histogram6,
  "image7": histogram7,
  "image8": histogram8,
  "image9": histogram9,
  "image10": histogram10,
}

thisdict = {}
def distances1(histog):
  i=0
  for cle,valeur in thisdicttexture.items():
    d= inter_distance(histog,valeur)
    thisdict["static/dataBase/image"+str(i)+".jpg"]=d
    i=i+1
  sorteddict= dict(sorted(thisdict.items(), key=lambda item: item[1]))
  n_items = take(5, sorteddict.keys())
  return n_items
'''
plt.figure(figsize=(10,8))
plt.subplot(231); plt.imshow(img1); plt.title('Query')
plt.subplot(232); plt.imshow(img3); plt.title('Rank 1 : %.4f' % dQ3_H)
plt.subplot(233); plt.imshow(img2); plt.title('Rank 2 : %.4f' % dQ2_H)
plt.subplot(234); plt.imshow(img4); plt.title('Rank 3 : %.4f' % dQ4_H)
plt.subplot(235); plt.imshow(img5); plt.title('Rank 4 : %.4f' % dQ5_H)
plt.subplot(236); plt.imshow(img6); plt.title('Rank 5 : %.4f' % dQ6_H)
plt.show()

vals = range(len(lbp1))
plt.figure(figsize=(10,5))
plt.subplot(231); plt.bar(vals,lbp1); 
plt.title('Query'); plt.axis('off')
plt.subplot(232); plt.bar(vals,lbp3); 
plt.title('Rank 1'); plt.axis('off')
plt.subplot(233); plt.bar(vals,lbp2); 
plt.title('Rank 2'); plt.axis('off')
plt.subplot(234); plt.bar(vals,lbp4); 
plt.title('Rank 3'); plt.axis('off')
plt.subplot(235); plt.bar(vals,lbp5); 
plt.title('Rank 4'); plt.axis('off')
plt.subplot(236); plt.bar(vals,lbp6); 
plt.title('Rank 5'); plt.axis('off')
plt.show()
  "image11": histogram11,
  "image12": histogram12,
  "image13": histogram13,
  "image14": histogram14,
  "image15": histogram15,
  "image16": histogram16,
  "image17": histogram17,
  "image18": histogram18,
  "image19": histogram19,
  "image20": histogram20,
  "image21": histogram21,
  "image22": histogram22,
  "image23": histogram23,
  "image24": histogram24,
  "image25": histogram25,
  "image26": histogram26,
  "image27": histogram27,
  "image28": histogram28,
  "image29": histogram29,
  "image30": histogram30,
  "image31": histogram31,
  "image32": histogram32,
  "image33": histogram33,
  "image34": histogram34,
  "image35": histogram35,
  "image36": histogram36,
  "image37": histogram37,
  "image38": histogram38,
  "image39": histogram39,
  "image40": histogram40,
  "image41": histogram41,
  "image42": histogram42,
  "image43": histogram43,
  "image44": histogram44,
  "image45": histogram45,
  "image46": histogram46,
  "image47": histogram47,
  "image48": histogram48,
  "image49": histogram49,
  "image50": histogram50,
  "image51": histogram51,
  "image52": histogram52,
  "image53": histogram53,
  "image54": histogram54,
  "image55": histogram55,
  "image56": histogram56,
  "image57": histogram57,
  "image58": histogram58,
  "image59": histogram59,
  "image60": histogram60,
  "image61": histogram61,
  "image62": histogram62,
  "image63": histogram63,
  "image64": histogram64,
  "image65": histogram65,
  "image66": histogram66,
  "image67": histogram67,
  "image68": histogram68,
  "image69": histogram69,
  "image70": histogram70,
  "image71": histogram71,
  "image72": histogram72,
  "image73": histogram73,
  "image74": histogram74,
  "image75": histogram75,
  "image76": histogram76,
  "image77": histogram77,
  "image78": histogram78,
  "image79": histogram79,
  "image80": histogram80,
  "image81": histogram81,
  "image82": histogram82,
  "image83": histogram83,
  "image84": histogram84,
  "image85": histogram85,
  "image86": histogram86,
  "image87": histogram87,
  "image88": histogram88,
  "image89": histogram89,
  "image90": histogram90,
  "image91": histogram91,
  "image92": histogram92,
  "image93": histogram93,
  "image94": histogram94,
  "image95": histogram95,
  "image96": histogram96,
  "image97": histogram97,
  "image98": histogram98,
  "image99": histogram99
  
'''