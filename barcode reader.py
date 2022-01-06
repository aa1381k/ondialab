import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os

   
#convert cv format to pil format
def convert_to_pil(cv_image):
    color_coverted = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image=Image.fromarray(color_coverted)
    return pil_image

#convert pil format to cv format
def convert_to_cv2(pil_image):
    numpy_image=np.array(pil_image)  
    opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGBA2BGR) 
    return opencv_image

#crop image func       
def crop(cr,left,up,right,down):
    w,h=cr.size # get image size
    if (up>300 and down>500) or (up==0 and down<500): #Halve or not?
        try:
            crop=cr.crop((left, up, right, down)) #yes
        except:
            print("error: The up must be 500 or 0 and down must be 480 or 900 ")
    else:
        crop=cr.crop((left, up, w-right, h-down)) #no
    return crop

#adjust image color
def adj_color(image,adj):

    converter = ImageEnhance.Color(image)
    img = converter.enhance(adj)
    img2=convert_to_cv2(img)
    return img2

#convert image to black and white image
def black_white(rgb_image):
    image_gray=cv2.cvtColor(rgb_image,cv2.COLOR_BGR2GRAY)
    return image_gray

#add white backgroun to image
def background(bg):
    img_w, img_h = bg.size
    background = Image.new('RGBA', (1440, 900), (255, 255, 255, 255))
    bg_w, bg_h = background.size
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    back=background.paste(bg, offset)
    background=convert_to_cv2(background)
    return background

#adjust image contrast func
def Contrast(image):
    image=convert_to_pil(image)
    enhancer = ImageEnhance.Contrast(image)
    factor = 100 #increase contrast
    im_output = enhancer.enhance(factor)  
    return im_output

def txt_file(text,data):

    if os.path.exists("data.txt")==False:
        file=open("data.txt","x")

    file=open("data.txt","a")
    file.write(text+data+"\n")

#zero or one func
def zero_one(image):
    
    path=os.getcwd()
    path=path+"/"+"Barcodes for decode"+"/"+image

    img2 = cv2.imread(path, cv2.IMREAD_COLOR) 
    
        # Reading same image in another variable and  
        # converting to gray scale. 
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        # Converting image to a binary image  
        # (black and white only image). 
        
    Binary=0
    _,threshold = cv2.threshold(img, 127,255,1, 
                                    cv2.THRESH_BINARY) 
        
        # Detecting shapes in image by selecting region  
        # with same colors or intensity. 
    contours,_=cv2.findContours(threshold, cv2.RETR_TREE, 
                            cv2.CHAIN_APPROX_SIMPLE) 

        # Searching through every region selected to  
        # find the required polygon. 
    for cnt in contours : 
        area = cv2.contourArea(cnt) 
            # Shortlisting the regions based on there area. 

        if area > 5000:  
            approx = cv2.approxPolyDP(cnt,  
                                0.009 * cv2.arcLength(cnt, True), True) 
        
                # Checking if the no. of sides of the selected region is 4. 
            if(len(approx) <5):  
                cv2.drawContours(img2, [approx], -1, (0, 0, 255), 2)
                Binary=1
                return Binary

        elif area <5000 and area>500:  
            approx = cv2.approxPolyDP(cnt,  
                                0.009 * cv2.arcLength(cnt, True), True) 
                    # Checking if the no. of sides of the selected region is 4. 
            if(len(approx) <5):  
                cv2.drawContours(img2, [approx], -1, (0, 255, 0), 2)
                Binary=0
                return Binary       

#detect rectangles
def rectangles (image):
    a=[]
    b=[]
    image1=image
    image_names=image
    image = cv2.imread(image)
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blurred, 120, 255, 1)
    kernel = np.ones((5,5),np.uint8)
    dilate = cv2.dilate(canny, kernel, iterations=1)
    img2 = cv2.imread(image1, cv2.IMREAD_COLOR) 
        
            # Reading same image in another variable and  
            # converting to gray scale. 
    img = cv2.imread(image1, cv2.IMREAD_GRAYSCALE) 
            # Converting image to a binary image  
            # (black and white only image). 
            

    _,threshold = cv2.threshold(img, 127,255,1, 
                                        cv2.THRESH_BINARY) 
            
            # Detecting shapes in image by selecting region  
            # with same colors or intensity. 
    contours,_=cv2.findContours(threshold, cv2.RETR_TREE, 
                                cv2.CHAIN_APPROX_SIMPLE) 
    # Find contours
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]


    for cnt in contours: 
            area = cv2.contourArea(cnt) 
                # Shortlisting the regions based on there area. 

            if area > 5000:  
                approx = cv2.approxPolyDP(cnt,  
                                    0.009 * cv2.arcLength(cnt, True), True) 
            
                    # Checking if the no. of sides of the selected region is 4. 
                if(len(approx) <5):  
                    a.append(cv2.drawContours(img2, [approx], -1, (0, 0, 255), 2))

            if area <5000 and area>500:  
                approx = cv2.approxPolyDP(cnt,  
                                    0.009 * cv2.arcLength(cnt, True), True) 
            
                    # Checking if the no. of sides of the selected region is 4. 
                if(len(approx) <5):  
                    b.append(cv2.drawContours(img2, [approx], -1, (0, 255, 0), 2))
  
    if image_names=="0.png":

        parent_dir=os.getcwd()#get file location

        if os.path.exists("Barcodes for decode")==False: #Check the existence of the folder

            directory="Barcodes for decode" #folder name
            # Path 
            path = os.path.join(parent_dir, directory) 
            # mode 
            mode = 0o666

            os.mkdir(path, mode) #make folder


        # Iterate thorugh contours and filter for ROI
        image_number = 0
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
            ROI = original[y:y+h, x:x+w]
            ROI=convert_to_pil(ROI)
            ROI_background=background(ROI) #add white background
            cv2.imwrite("{}/Barcodes for decode/image_{}.png".format(parent_dir,image_number),ROI_background) #save image to Barcodes for decode folder
            image_number += 1
        print("\n--------barcode unit-----------")
        print("Large rectangle: %s" %len(a)) #print number of large barcode in cmd
        txt_file("Large rectangle: %s" %len(a),"") #write number of large barcode in text file

        print("Small rectangle: %s" %len(b)) #print number of small barcode in cmd
        txt_file("Small rectangle: %s" %len(b),"") #write number of small barcode in text file

        LEN=(len(a)+len(b))
        if LEN<12:
            print("The number of barcodes should be 12, but it is {}".format(LEN))
    else:
        print("-----------strip unit---------")
        print("Number of strips: %s"%(len(a)+len(b)))
        txt_file("Number of strips: %s"%(len(a)+len(b)),"") #write number of strips in text file

    return img2

#convert binary to decimal func
def binary( list_name ):
    my_lst_str = ''.join(map(str, list_name))
    decimal=int(my_lst_str, 2)
    return decimal

# decode side,bits,control line
def decode():

    side_shape=[]
    bits=[]
    ############side
    for i in range(0,2):
        Bool=zero_one("image_{}.png".format(i)) #check barcode

        if Bool==1:
            side_shape.append(1)
        else:
            side_shape.append(0)
    side_shape.reverse()# revers shape_list
    side=binary(side_shape) #convert binary to decimal

    #############control line
    print("-----------decode----------")
    Bool=zero_one("image_2.png") #check barcode
    if Bool==1:
        print("control line: 1")
        txt_file("control line= ",str(Bool))
    else:
        print("control line: 0")
        txt_file("control line= ",str(Bool))
        
    #############bits
    for i in range(3,11):
        Bool=zero_one("image_{}.png".format(i)) #check barcode

        if Bool==1:
            bits.append(1)
        else:
            bits.append(0)
    bits.reverse()# revers bits
    bits_decimal=binary(bits) #convert binary to decimal
    bits = ''.join(map(str, bits)) #convert list to str
    print("bits decimal: ",bits_decimal)
    txt_file("bits decimal: ",str(bits_decimal))
    if len(bits)>3:
        print("bits binary: ",bits) #print bits

    # detect image side
    if side==0:
       print("Side=Dimond")
       txt_file("Side=Dimond","")
    elif side==1:
       print("Side=Triangle") 
       txt_file("Side=Triangle","")
    elif side==2:
       print("Side=Disk")
       txt_file("Side=Disk","") 
    elif side==3:
       print("Side=Cross")
       txt_file("Side=Cross","")
 
    txt_file("bits binary:",bits)


while True:

        
    image_name=input("enter image number: ") #input image number
    txt_file("-------"+image_name,".jpg"+"-------")
    image=Image.open(image_name+".jpg") # open image
    image=adj_color(image,18) #adj image color
    image_denoising=cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21) #denoising
    image_gray=cv2.cvtColor(image_denoising,cv2.COLOR_BGR2GRAY) #black and white image
    image_contrast=Contrast(image_gray)#adj contrast
    image_crop=crop(image_contrast,30,5,30,30)#crop image
    image_resize=image_crop.resize((300, 700)) #resize image


    barcode_unit=crop(image_resize,0,0,300,380) #crop barcode unit
    barcode_back=background(barcode_unit)# add white background


    strip_unit=crop(image_resize,0,430,300,670) #crop strip unit
    strip_back=background(strip_unit)# add white background

    image_unit=[barcode_back,strip_back]

    for i in range(2):
        cv2.imwrite("%s.png"%i,image_unit[i])

    barcode=rectangles("0.png")
    strip=rectangles("1.png")

    path=os.getcwd()#get file location
    file_list=os.listdir(path+"\Barcodes for decode")#List of files

    decode()

    cv2.imshow("barcode",barcode)
    cv2.imshow("strip",strip)
    cv2.waitKey(0)
    cv2.destroyAllWindows

        
