import cv2
import numpy as np
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
from scipy import ndimage
from scipy import stats
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QGraphicsDropShadowEffect


# استيراد ملف الواجهة الرسومية
FORM_CLASS, _ = loadUiType(r"d:\PYthon Projects\image processing project\main.ui")


class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.setupUi(self)  # تحميل الواجهة من ملف UI

        shadow1 = QGraphicsDropShadowEffect()
        shadow1.setBlurRadius(30)         # كل ما زودت الرقم ده، الشادو يكون أنعم وانتشاره أكبر
        shadow1.setXOffset(0)             # صفر علشان الشادو يطلع من كل الاتجاهات
        shadow1.setYOffset(0)
        shadow1.setColor(QColor(0, 0, 0, 180))  # لون الشادو ودرجة الشفافية
        self.groupBox.setGraphicsEffect(shadow1)

        shadow3 = QGraphicsDropShadowEffect()
        shadow3.setBlurRadius(30)         # كل ما زودت الرقم ده، الشادو يكون أنعم وانتشاره أكبر
        shadow3.setXOffset(0)             # صفر علشان الشادو يطلع من كل الاتجاهات
        shadow3.setYOffset(0)
        shadow3.setColor(QColor(0, 0, 0, 180))  # لون الشادو ودرجة الشفافية
        self.groupBox_2.setGraphicsEffect(shadow3)
        
        

        shadow2 = QGraphicsDropShadowEffect()
        shadow2.setBlurRadius(30)         # كل ما زودت الرقم ده، الشادو يكون أنعم وانتشاره أكبر
        shadow2.setXOffset(0)             # صفر علشان الشادو يطلع من كل الاتجاهات
        shadow2.setYOffset(0)
        shadow2.setColor(QColor(0, 0, 0, 180))  # لون الشادو ودرجة الشفافية
        self.groupBox_4.setGraphicsEffect(shadow2)
        
        shadow5 = QGraphicsDropShadowEffect()
        shadow5.setBlurRadius(30)         # كل ما زودت الرقم ده، الشادو يكون أنعم وانتشاره أكبر
        shadow5.setXOffset(0)             # صفر علشان الشادو يطلع من كل الاتجاهات
        shadow5.setYOffset(0)
        shadow5.setColor(QColor(0, 0, 0, 180))  # لون الشادو ودرجة الشفافية
        self.groupBox_5.setGraphicsEffect(shadow5)
        
        shadow6 = QGraphicsDropShadowEffect()
        shadow6.setBlurRadius(30)         # كل ما زودت الرقم ده، الشادو يكون أنعم وانتشاره أكبر
        shadow6.setXOffset(0)             # صفر علشان الشادو يطلع من كل الاتجاهات
        shadow6.setYOffset(0)
        shadow6.setColor(QColor(0, 0, 0, 180))  # لون الشادو ودرجة الشفافية
        self.groupBox_6.setGraphicsEffect(shadow6)
        
        # button.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        # _______________________________________________________buttom page one __________________________________________
        self.pushButton_3.clicked.connect(lambda: self.go_to_page(1))  
        self.pushButton_4.clicked.connect(lambda: self.go_to_page(2)) 
        self.pushButton.clicked.connect(lambda: self.go_to_page(0)) 
        self.pushButton_2.clicked.connect(lambda: self.go_to_page(0)) 
        
        # __________________________________________________________buttom page two________________________________________
        self.buttom_load_imageone_3.clicked.connect(self.load_image)  
        self.buttom_additionimageone_3.clicked.connect(self.addition) 
        self.buttom_subtractionimageone_3.clicked.connect(self.subtraction)
        self.buttom_divisionimageone_3.clicked.connect(self.division)
        self.buttom_complementimageone_3.clicked.connect(self.complement)
        self.buttom_change_red_3.clicked.connect(self.Change_Red)
        self.buttom_Swap_R_and_G_channels_3.clicked.connect(self.Swap_R_and_G_channels)
        self.buttom_Eliminate_Red_3.clicked.connect(self.Eliminate_Red)
        self.pushButton_8.clicked.connect(self.histogram_stretching)
        self.pushButton_9.clicked.connect(self.Equalized_Image)
        self.pushButton_38.clicked.connect(self.apply_average_filter_neighborhood)
        self.pushButton_40.clicked.connect(self.apply_laplacian_filter)
        self.pushButton_37.clicked.connect(self.apply_maximum_filter)
        self.pushButton_39.clicked.connect(self.apply_minimum_filter)
        self.pushButton_27.clicked.connect(self.apply_median_filter)
        # self.pushButton_15.clicked.connect(self.mode_filter)
        self.pushButton_65.clicked.connect(self.apply_average_filter_restoration)
        self.buttom_101.clicked.connect(self.median_filter_restoration)
        # self.buttom_outlier_filter_4.clicked.connect(self.outlier_method)
        self.pushButton_61.clicked.connect(self.average_filter_to_gaussian_noise)
        self.pushButton_57.clicked.connect(self.apply_adaptive_threshold)
        self.pushButton_60.clicked.connect(self.otsu_thresholding)
        self.pushButton_64.clicked.connect(self.Basic_Global_Thresholding)
        self.pushButton_58.clicked.connect(self.Sobel_detector)
        self.pushButton_63.clicked.connect(self.dilation_operation)
        self.pushButton_62.clicked.connect(self.erosion_operation)
        self.pushButton_59.clicked.connect(self.opening_operation)
        

        
        # __________________________________________________________buttom page three________________________________________
        
        
        self.buttom_load_imageone_8.clicked.connect(self.load_image1)
        self.buttom_load_imageone_7.clicked.connect(self.load_image2)
        self.buttom_addition_two_imge_3.clicked.connect(self.add_images)
        self.buttom_subtraction_two_imagee_3.clicked.connect(self.subtract_images)
        self.buttom_division_two_image_3.clicked.connect(self.divide_images)
        
        
        # __________________________________________________________ main function ________________________________________
    def load_image(self):
        """ دالة لتحميل الصورة من الجهاز وعرضها في label image_input """
        file_name, _ = QFileDialog.getOpenFileName(self, "اختيار صورة", "", "Image Files (*.png *.jpg *.bmp *.jpeg)")
        if file_name:  
            self.img = cv2.imread(file_name)  
            self.display_image(self.img, self.image_input)  
            
            
    def display_image(self, img, label):
        qformat = QImage.Format_Indexed8
    
        if len(img.shape) == 3:  # صورة ملونة
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:  # صورة رمادية
            qformat = QImage.Format_Grayscale8
    
        height, width = img.shape[:2]
        bytes_per_line = 3 * width if len(img.shape) == 3 else width
    
        qimg = QImage(img.data, width, height, bytes_per_line, qformat)
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap)
        label.setScaledContents(True)
    def confrim_load_image(self):
        """ دالة لزيادة سطوع الصورة وعرضها في QLabel image_output """
        if self.img is None:
            QMessageBox.warning(self, "تحذير", "يرجى تحميل صورة أولًا!")
            return
    def go_to_page(self,number_page):
        # الانتقال إلى التبويبة رقم 1 (يعني التانية)
        self.tabWidget.setCurrentIndex(number_page)
        
        # __________________________________________________________ function page one ________________________________________
    def addition(self):
        self.confrim_load_image

        output=cv2.add(self.img,50)
        
        self.display_image(output, self.image_output) 

    def subtraction(self):
        self.confrim_load_image
        output=cv2.subtract(self.img,50)
    
        self.display_image(output, self.image_output)    
        

    def division(self):
        self.confrim_load_image
    
        output=cv2.divide(self.img,3)
    
        self.display_image(output, self.image_output)   

    def complement(self):
        self.confrim_load_image
    
        output=255-self.img
    
        self.display_image(output, self.image_output)   
        

    def Change_Red(self):
        self.confrim_load_image
        
        img_red = self.img.copy()
        img_red[:, :, 2] = 255
      
    
        self.display_image( img_red, self.image_output)  

    def Swap_R_and_G_channels(self):
        
        self.confrim_load_image
        img_swap = self.img.copy()
        temp = img_swap[:, :, 1].copy()
        img_swap[:, :, 1] = img_swap[:, :, 2]
        img_swap[:, :, 2] = temp

        self.display_image( img_swap, self.image_output)  


    def Eliminate_Red(self):
        
        self.confrim_load_image
        img_no_red = self.img.copy()
        img_no_red[:, :, 2] = 0
        self.display_image( img_no_red, self.image_output)  
 
 
     
    def histogram_stretching(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import cv2
    
        # تحويل الصورة لرمادي
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # self.display_image(gray_img, self.image_input)
    
        # حساب أقل وأعلى قيمة
        min_val = np.min(gray_img)
        max_val = np.max(gray_img)
    
        # تجنب القسمة على صفر
        if max_val - min_val == 0:
            self.display_image(gray_img, self.image_output)
            return
    
        # تطبيق Histogram Stretching
        stretched = ((gray_img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        self.display_image(stretched, self.image_output)
    
        # عرض الصور والهستوجرامات
        plt.figure(figsize=(10, 6))
    
        plt.subplot(2, 2, 1)
        plt.imshow(self.img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
    
        plt.subplot(2, 2, 2)
        plt.imshow(stretched, cmap='gray')
        plt.title('Stretched Image')
        plt.axis('off')
    
        plt.subplot(2, 2, 3)
        plt.hist(self.img.ravel(), bins=256, range=(0, 256), color='gray')
        plt.title('Original Histogram')
    
        plt.subplot(2, 2, 4)
        plt.hist(stretched.ravel(), bins=256, range=(0, 256), color='gray')
        plt.title('Stretched Histogram')
    
        plt.tight_layout()
        plt.show()

        return 
    
    def Equalized_Image(self):
        
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        
        # تطبيق Histogram Equalization
        equalized = cv2.equalizeHist(gray_img)
        self.display_image(equalized, self.image_output)
        
        # عرض النتائج
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 2, 1)
        plt.imshow(gray_img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(equalized, cmap='gray')
        plt.title('Equalized Image')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.hist(gray_img.ravel(), bins=256, range=(0, 256), color='gray')
        plt.title('Original Histogram')
        
        plt.subplot(2, 2, 4)
        plt.hist(equalized.ravel(),bins= 256,range= (0, 256), color='gray')
        plt.title('Equalized Histogram')
        
        plt.tight_layout()
        plt.show(block=False)
# ________________________________________________________________________________________________________
       
      
       
    
    def apply_average_filter_neighborhood(self):
      
      kernel1 = np.ones((3,3),np.float32)/9
      average=cv2.filter2D(self.img, -1,kernel1)
      self.display_image(average , self.image_output)  
       
    
        


  
    def apply_laplacian_filter(self):
 

    
     
        kernel = np.array([[1, -2, 1],
                           [-2, 4, -2],
                           [1, -2, 1]])
    
        # تطبيق الفلتر
        laplacian = cv2.filter2D(self.img, -1, kernel)
    
        # عرض الصورة بعد الفلتر
        self.display_image(laplacian, self.image_output)
    
    
    def apply_maximum_filter(self):
        self.confrim_load_image
        channels = cv2.split(self.img)
        filtered = [ndimage.maximum_filter(ch, size=3) for ch in channels]
        result= cv2.merge(filtered)
        
        self.display_image( result, self.image_output)
    
    def apply_minimum_filter(self):
        self.confrim_load_image
        channels = cv2.split(self.img)
        filtered = [ndimage.minimum_filter(ch, size=3) for ch in channels]
        result= cv2.merge(filtered)
        self.display_image( result, self.image_output)  

    def apply_median_filter(self):  
           
        self.confrim_load_image
        channels = cv2.split(self.img)
        filtered = [ndimage.median_filter(ch, size=3) for ch in channels]
        result= cv2.merge(filtered)     
        self.display_image( result, self.image_output)  
       
    
        
 

    def mode_filter(self ):
        pass 
    
    # _____________________________________________________
    def add_salt_and_pepper_noise(self):
        prob = 0.04
        noisy = np.copy(self.img)
    
        thres = 1 - prob
    
        if len(noisy.shape) == 3:  # صورة ملونة
            for i in range(noisy.shape[0]):
                for j in range(noisy.shape[1]):
                    rnd = np.random.rand()
                    if rnd < prob:
                        noisy[i, j] = [0, 0, 0]
                    elif rnd > thres:
                        noisy[i, j] = [255, 255, 255]
        else:  # صورة رمادية
            for i in range(noisy.shape[0]):
                for j in range(noisy.shape[1]):
                    rnd = np.random.rand()
                    if rnd < prob:
                        noisy[i, j] = 0
                    elif rnd > thres:
                        noisy[i, j] = 255
    
        self.noisy = noisy  # الاسم الصح
        self.display_image(noisy, self.image_input)
        print("✅ Salt & Pepper noise added")

    def apply_average_filter_restoration(self):
        self.add_salt_and_pepper_noise()
        kernel = np.ones((3,3),np.float32)/9

        average=cv2.filter2D(self.noisy , -1,kernel)

        self.display_image( average , self.image_output)  
        
    def median_filter_restoration(self):
        self.add_salt_and_pepper_noise()

        filter=ndimage.median_filter(self.noisy ,size= 3)
        self.display_image( filter , self.image_output)  
    
    def outlier_method():
        print("ahmed morsi")
    
    # _________________Gaussian noise in python__________________________________
    def add_gaussian_noise( self ):
        mean=0
        var=0.01
        row, col, ch = self.img.shape
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch)) * 255
        noisy = self.img + gauss
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        self.noisy1=noisy
        self.display_image( noisy , self.image_input)  
       
    
    def average_filter_to_gaussian_noise(self):
        mean = 0
        sigma = 25
        gaussian_noise = np.random.normal(mean, sigma, self.img.shape)
        noisy_image = np.uint8(np.clip(self.img + gaussian_noise, 0, 255))
        
        # تعريف الفلتر المتوسط
        kernel = np.ones((3, 3), np.float32) / 9  # فلتر متوسط 3x3
        
        # تطبيق الفلتر على الصورة المشوشة
        filtered_image = cv2.filter2D(noisy_image, -1, kernel)

        self.display_image( filtered_image , self.image_output)  
    
        # __________________________________________________________function page two ________________________________________
    def apply_adaptive_threshold(self):
         # قراءة الصورة باللون الرمادي
         image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
         self.display_image( image , self.image_input)  
         binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
         cv2. THRESH_BINARY_INV, 199, 5) 

         self.display_image( binary , self.image_output)  
        
        
        
    def otsu_thresholding(self):
        # قراءة الصورة باللون الرمادي
    
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.display_image( img , self.image_input)  
        
        # تطبيق Otsu's Thresholding
        _, im_bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.display_image( im_bw , self.image_output)  
    


    def Basic_Global_Thresholding(self):
       
       threshold_value = 128
       
       _, thresh_basic = cv2.threshold(self.img, threshold_value, 255, cv2.THRESH_BINARY)

       self.display_image(thresh_basic  , self.image_output)  

    

    #    _______________________________________________________________________________________________-

    def Sobel_detector(self):
       
        # تطبيق Sobel في الاتجاه X و Y
       sobelx = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=3)  # مشتقة في اتجاه X
       sobely = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=3)  # مشتقة في اتجاه Y
       
       # حساب الحجم النهائي للحواف
       sobel_combined = np.sqrt(sobelx**2 + sobely**2)
       sobel_combined = np.uint8(sobel_combined / np.max(sobel_combined) * 255)

       self.display_image( sobel_combined , self.image_output)  
# _____________________________________________________________________
    def dilation_operation(self):
    
        mask = np.ones((3, 3), np.uint8)
    
        result= cv2.dilate(self.img, mask, iterations=1)
        self.display_image( result , self.image_output)  
    
    
    def erosion_operation(self):
    
        mask = np.ones((3, 3), np.uint8)
    
        result= cv2.erode(self.img, mask, iterations=1)
        self.display_image( result , self.image_output)  
    
    def opening_operation(self):
    
        mask = np.ones((3, 3), np.uint8)
    
        result= cv2.morphologyEx(self.img,  cv2.MORPH_OPEN, mask)
        self.display_image( result , self.image_output)  
    
   # __________________________________________________________ function page three ________________________________________
    
    def load_image1(self):
        """ دالة لتحميل الصورة من الجهاز وعرضها في label image_input """
        file_name, _ = QFileDialog.getOpenFileName(self, "اختيار صورة", "", "Image Files (*.png *.jpg *.bmp *.jpeg)")
        if file_name:  
            self.img1 = cv2.imread(file_name)  
            self.display_image(self.img1, self.img1_input_2)  
            
    def load_image2(self):
        """ دالة لتحميل الصورة من الجهاز وعرضها في label image_input """
        file_name, _ = QFileDialog.getOpenFileName(self, "اختيار صورة", "", "Image Files (*.png *.jpg *.bmp *.jpeg)")
        if file_name:  
            self.img2 = cv2.imread(file_name)  
            self.display_image(self.img2, self.image_input_9)  
            
               
    
        
        
    def add_images(self):
        # الحجم المطلوب
        target_width = 431
        target_height = 501
    
        # تغيير حجم الصورتين إلى الحجم المطلوب
        self.img1 = cv2.resize(self.img1, (target_width, target_height))
        self.img2 = cv2.resize(self.img2, (target_width, target_height))
        self.display_image(self.img2, self.image_input_9)  
        self.display_image(self.img1, self.img1_input_2)  
        
        # جمع الصورتين
        added_image = cv2.add(self.img1, self.img2)
    
        # عرض الصورة الناتجة
        self.display_image(added_image, self.ouput2image_2)
           
    def subtract_images(self):
         # التأكد إن الصورتين بنفس الحجم
         if self.img1.shape != self.img2.shape:
        
             img2 = cv2.resize(self.img2, (self.img1.shape[1],self.img1.shape[0]))
             self.img2=img2
     
         # جمع الصورتين باستخدام OpenCV (مع الحفاظ على الحدود)
         subtract_image = cv2.subtract(self.img1, self.img2)
         self.display_image( subtract_image , self.ouput2image_2)  
         
    def divide_images(self):

        # التأكد من وجود الصورتين
        if self.img1 is None or self.img2 is None:
            raise ValueError("في مشكلة في قراءة الصور")
    
        # تحديد الحجم المطلوب (عرض × طول)
        target_size = (431, 501)
    
        # تغيير حجم الصورتين
        img1_resized = cv2.resize(self.img1, target_size)
        img2_resized = cv2.resize(self.img2, target_size)
    
        # تحويلهم لـ float لتجنب القسمة على صفر
        img1_float = img1_resized.astype(np.float32)
        img2_float = img2_resized.astype(np.float32)
    
        # إضافة قيمة صغيرة لتجنب القسمة على صفر
        epsilon = 1e-5
        divided = cv2.divide(img1_float, img2_float + epsilon)
    
        # تطبيع الناتج وتحويله لـ uint8 للعرض
        result = cv2.normalize(divided, None, 0, 255, cv2.NORM_MINMAX)
        result = result.astype(np.uint8)
    
        # عرض الصورة الناتجة
        self.display_image(result, self.ouput2image_2)




def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()