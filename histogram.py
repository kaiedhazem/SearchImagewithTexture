import cv2

# fonction de calcul des histogrammes aprés la transformation en niveau de gris
def histog(imag):
 gray_image = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
 histog = cv2.calcHist([gray_image], [0], None, [256], [0, 256])	
 return histog

# fonction de calcul des distances d'intersection
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
image0 = cv2.imread('static/dataBase/image0.jpg')
image1 = cv2.imread('static/dataBase/image1.jpg')
image2 = cv2.imread('static/dataBase/image2.jpg')
image3 = cv2.imread('static/dataBase/image3.jpg')
image4 = cv2.imread('static/dataBase/image4.jpg')
image5 = cv2.imread('static/dataBase/image5.jpg')
image6 = cv2.imread('static/dataBase/image6.jpg')
image7 = cv2.imread('static/dataBase/image7.jpg')
image8 = cv2.imread('static/dataBase/image8.jpg')
image9 = cv2.imread('static/dataBase/image9.jpg')
image10 = cv2.imread('static/dataBase/image10.jpg')
image11 = cv2.imread('static/dataBase/image11.jpg')
image12 = cv2.imread('static/dataBase/image12.jpg')
image13 = cv2.imread('static/dataBase/image13.jpg')
image14 = cv2.imread('static/dataBase/image14.jpg')
image15 = cv2.imread('static/dataBase/image15.jpg')
image16 = cv2.imread('static/dataBase/image16.jpg')
image17 = cv2.imread('static/dataBase/image17.jpg')
image18 = cv2.imread('static/dataBase/image18.jpg')
image19 = cv2.imread('static/dataBase/image19.jpg')
image20= cv2.imread('static/dataBase/image20.jpg')
image21 = cv2.imread('static/dataBase/image21.jpg')
image22 = cv2.imread('static/dataBase/image22.jpg')
image23 = cv2.imread('static/dataBase/image23.jpg')
image24 = cv2.imread('static/dataBase/image24.jpg')
image25 = cv2.imread('static/dataBase/image25.jpg')
image25 = cv2.imread('static/dataBase/image25.jpg')
image26 = cv2.imread('static/dataBase/image26.jpg')
image27 = cv2.imread('static/dataBase/image27.jpg')
image28 = cv2.imread('static/dataBase/image28.jpg')
image29 = cv2.imread('static/dataBase/image29.jpg')
image30 = cv2.imread('static/dataBase/image30.jpg')
image31 = cv2.imread('static/dataBase/image31.jpg')
image32 = cv2.imread('static/dataBase/image32.jpg')
image33 = cv2.imread('static/dataBase/image33.jpg')
image34 = cv2.imread('static/dataBase/image34.jpg')
image35 = cv2.imread('static/dataBase/image35.jpg')
image36 = cv2.imread('static/dataBase/image36.jpg')
image37 = cv2.imread('static/dataBase/image37.jpg')
image38 = cv2.imread('static/dataBase/image38.jpg')
image39 = cv2.imread('static/dataBase/image39.jpg')
image40 = cv2.imread('static/dataBase/image40.jpg')
image41 = cv2.imread('static/dataBase/image41.jpg')
image42 = cv2.imread('static/dataBase/image42.jpg')
image43 = cv2.imread('static/dataBase/image43.jpg')
image44 = cv2.imread('static/dataBase/image44.jpg')
image45 = cv2.imread('static/dataBase/image45.jpg')
image46 = cv2.imread('static/dataBase/image46.jpg')
image47 = cv2.imread('static/dataBase/image47.jpg')
image48 = cv2.imread('static/dataBase/image48.jpg')
image49 = cv2.imread('static/dataBase/image49.jpg')
image50 = cv2.imread('static/dataBase/image50.jpg')
image51 = cv2.imread('static/dataBase/image51.jpg')
image52 = cv2.imread('static/dataBase/image52.jpg')
image53 = cv2.imread('static/dataBase/image53.jpg')
image54 = cv2.imread('static/dataBase/image54.jpg')
image55 = cv2.imread('static/dataBase/image55.jpg')
image56 = cv2.imread('static/dataBase/image56.jpg')
image57 = cv2.imread('static/dataBase/image57.jpg')
image58 = cv2.imread('static/dataBase/image58.jpg')
image59 = cv2.imread('static/dataBase/image59.jpg')
image60 = cv2.imread('static/dataBase/image60.jpg')
image61 = cv2.imread('static/dataBase/image61.jpg')
image62 = cv2.imread('static/dataBase/image62.jpg')
image63 = cv2.imread('static/dataBase/image63.jpg')
image64 = cv2.imread('static/dataBase/image64.jpg')
image65 = cv2.imread('static/dataBase/image65.jpg')
image66 = cv2.imread('static/dataBase/image66.jpg')
image67 = cv2.imread('static/dataBase/image67.jpg')
image68 = cv2.imread('static/dataBase/image68.jpg')
image69 = cv2.imread('static/dataBase/image69.jpg')
image70 = cv2.imread('static/dataBase/image70.jpg')
image71 = cv2.imread('static/dataBase/image71.jpg')
image72 = cv2.imread('static/dataBase/image72.jpg')
image73 = cv2.imread('static/dataBase/image73.jpg')
image74 = cv2.imread('static/dataBase/image74.jpg')
image75 = cv2.imread('static/dataBase/image75.jpg')
image76 = cv2.imread('static/dataBase/image76.jpg')
image77 = cv2.imread('static/dataBase/image77.jpg')
image78 = cv2.imread('static/dataBase/image78.jpg')
image79 = cv2.imread('static/dataBase/image79.jpg')
image80 = cv2.imread('static/dataBase/image80.jpg')
image81 = cv2.imread('static/dataBase/image81.jpg')
image82 = cv2.imread('static/dataBase/image82.jpg')
image83 = cv2.imread('static/dataBase/image83.jpg')
image84 = cv2.imread('static/dataBase/image84.jpg')
image85 = cv2.imread('static/dataBase/image85.jpg')
image86 = cv2.imread('static/dataBase/image86.jpg')
image87 = cv2.imread('static/dataBase/image87.jpg')
image88 = cv2.imread('static/dataBase/image88.jpg')
image89 = cv2.imread('static/dataBase/image89.jpg')
image90 = cv2.imread('static/dataBase/image90.jpg')
image91 = cv2.imread('static/dataBase/image91.jpg')
image92 = cv2.imread('static/dataBase/image92.jpg')
image93 = cv2.imread('static/dataBase/image93.jpg')
image94 = cv2.imread('static/dataBase/image94.jpg')
image95 = cv2.imread('static/dataBase/image95.jpg')
image96 = cv2.imread('static/dataBase/image96.jpg')
image97 = cv2.imread('static/dataBase/image97.jpg')
image98 = cv2.imread('static/dataBase/image98.jpg')
image99 = cv2.imread('static/dataBase/image99.jpg')

#calcul des histogrammes 
histogram0 = histog(image0)
histogram1 = histog(image1)
histogram2 = histog(image2)
histogram3 = histog(image3)
histogram4 = histog(image4)
histogram5 = histog(image5)
histogram6 = histog(image6)
histogram7 = histog(image7)
histogram8 = histog(image8)
histogram9 = histog(image9)
histogram10 = histog(image10)
histogram11 = histog(image11)
histogram12 = histog(image12)
histogram13 = histog(image13)
histogram14 = histog(image14)
histogram15 = histog(image15)
histogram16 = histog(image16)
histogram17 = histog(image17)
histogram18 = histog(image18)
histogram19 = histog(image19)
histogram20 = histog(image20)
histogram21 = histog(image21)
histogram22 = histog(image22)
histogram23 = histog(image23)
histogram24 = histog(image24)
histogram25 = histog(image25)
histogram26 = histog(image26)
histogram27 = histog(image27)
histogram28 = histog(image28)
histogram29 = histog(image29)
histogram30 = histog(image30)
histogram31 = histog(image31)
histogram32 = histog(image32)
histogram33 = histog(image33)
histogram34 = histog(image34)
histogram35 = histog(image35)
histogram36 = histog(image36)
histogram37 = histog(image37)
histogram38 = histog(image38)
histogram39 = histog(image39)
histogram40 = histog(image40)
histogram41 = histog(image41)
histogram42 = histog(image42)
histogram43 = histog(image43)
histogram44 = histog(image44)
histogram45 = histog(image45)
histogram46 = histog(image46)
histogram47 = histog(image47)
histogram48 = histog(image48)
histogram49 = histog(image49)
histogram50 = histog(image50)
histogram51 = histog(image51)
histogram52 = histog(image52)
histogram53 = histog(image53)
histogram54 = histog(image54)
histogram55 = histog(image55)
histogram56 = histog(image56)
histogram57 = histog(image57)
histogram58 = histog(image58)
histogram59 = histog(image59)
histogram60 = histog(image60)
histogram61 = histog(image61)
histogram62 = histog(image62)
histogram63 = histog(image63)
histogram64 = histog(image64)
histogram65 = histog(image65)
histogram66 = histog(image66)
histogram67 = histog(image67)
histogram68 = histog(image68)
histogram69 = histog(image69)
histogram70 = histog(image70)
histogram71 = histog(image71)
histogram72 = histog(image72)
histogram73 = histog(image73)
histogram74 = histog(image74)
histogram75 = histog(image75)
histogram76 = histog(image76)
histogram77 = histog(image77)
histogram78 = histog(image78)
histogram79 = histog(image79)
histogram80 = histog(image80)
histogram81 = histog(image81)
histogram82 = histog(image82)
histogram83 = histog(image83)
histogram84 = histog(image84)
histogram85 = histog(image85)
histogram86 = histog(image86)
histogram87 = histog(image87)
histogram88 = histog(image88)
histogram89 = histog(image89)
histogram90 = histog(image90)
histogram91 = histog(image91)
histogram92 = histog(image92)
histogram93 = histog(image93)
histogram94 = histog(image94)
histogram95 = histog(image95)
histogram96 = histog(image96)
histogram97 = histog(image97)
histogram98 = histog(image98)
histogram99 = histog(image99)
'''
#calcul des distances d'intersection entre image requête (green.jpg) et les autres images
d1= inter_distance(histogram,histogram1)
d2= inter_distance(histogram,histogram2)
d3= inter_distance(histogram,histogram3)
d4= inter_distance(histogram,histogram4)
d5= inter_distance(histogram,histogram5)
d6= inter_distance(histogram,histogram6)
d7= inter_distance(histogram,histogram7)
d8= inter_distance(histogram,histogram8)
d9= inter_distance(histogram,histogram9)
'''
# stockage des distances dans un dictionnaire python pour trier 
thisdictcolor = {
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
}
thisdict = {}
i=0
def distances(histog):
  for clef,v in thisdictcolor:
    d= inter_distance(histog,v)
    thisdict["image"+i]=d
    i=i+1
  sorteddict= dict(sorted(thisdict.items(), key=lambda item: item[1]))
  return sorteddict
'''
# affichage de resultat de plus proche jusqu'a plus loin selon distance d'intersection
print ("liste des images ordonnées de plus proche image jusqu'a la plus loin image selon la distance d'intersection :")
for key, value in sorteddict.items() :
    print (key)
'''