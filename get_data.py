# python3 script to extract data from the Images

# Import required packages 
import cv2
import os
import pytesseract 
import csv

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class ExtractData():
    """ Extraxt Data from the Images
        
    """
    def __init__(self):
        
        self.folder = "image_data"
    
    def get_data_from_images(self):
        
        """ read imges from the folder and 
            extract the data and store into csv files 
            
        """
        
        # Read image from the folder
        for filename in os.listdir(self.folder):
            all_data = []
            img = cv2.imread(os.path.join(self.folder,filename))
            if img is not None:

                # Convert the image to gray scale 
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
                # Performing OTSU threshold 
                ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 

                # Specify structure shape and kernel size. 
                # Kernel size increases or decreases the area 
                # of the rectangle to be detected. 
                # A smaller value like (10, 10) will detect 
                # each word instead of a sentence. 
                rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 

                # Appplying dilation on the threshold image 
                dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 

                # Finding contours 
                _, contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
                                                                cv2.CHAIN_APPROX_NONE) 

                # Looping through the identified contours 
                # Then rectangular part is cropped and passed on 
                # to pytesseract for extracting text from it 
                for cnt in contours: 
                    x, y, w, h = cv2.boundingRect(cnt) 

                    # Drawing a rectangle on image 
                    rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) 

                    # Cropping the text block for giving input to OCR 
                    cropped = img[y:y + h, x:x + w]  

                    # Apply OCR on the cropped image 
                    text = pytesseract.image_to_string(cropped)
                    list_test = [text]
                    
                    # clean the text
                    list2 = [x.replace('\n', '').replace('  ', '').replace('\x0c', '') for x in list_test]
                    all_data.append(list2)

                # store data into csv
                with open(filename+".csv", 'w') as myfile:
                    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    for d in all_data:
                        wr.writerow(d)
                    myfile.close()
        return "Image Data Extracted sucessfully"
            
            
if __name__ == '__main__':
    results = ExtractData().get_data_from_images()
    print(results)
    
