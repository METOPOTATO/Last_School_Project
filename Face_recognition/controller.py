import numpy as np 
import keras
from keras.models import load_model
import cv2 
import sys 

from model import DB

class Controller:
    def __init__(self):

        self.model = load_model('lib/facenet_keras.h5')

        self.cascade = cv2.CascadeClassifier('lib/haarcascade_frontalface_default.xml')
        self.mask_model = load_model('lib/model_mask_9954.h5')
        self.db = DB()

    def extract_face(self,img,size = (160,160)):
        try:
            height,width = img.shape[:2]

            modelFile = "lib/opencv_face_detector_uint8.pb"
            configFile = "lib/opencv_face_detector.pbtxt"
            net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

            blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)

            net.setInput(blob)
            detections = net.forward()
            confidence = detections[0, 0, 0, 2]
            if confidence > 0.2:
                x1 = int(detections[0, 0, 0, 3] * width)
                y1 = int(detections[0, 0, 0, 4] * height)
                x2 = int(detections[0, 0, 0, 5] * width)
                y2 = int(detections[0, 0, 0, 6] * height)

                location = x1,y1,x2-x1,y2-y1  
                face =img[y1:y2,x1:x2]
                face = cv2.resize(face,size)
                # cv2.imshow('face',face)
            else:
                return None
        except Exception as e:
            print(e)
            return None
        return face, location

    def get_embedding(self,face):
        face = face.astype('float32')
        mean,std =  face.mean(), face.std()
        standardized = (face - mean) / std
        samples = np.expand_dims(standardized,axis=0)
        predict = self.model.predict(samples)[0]
        return predict

    def load_data(self):
        id = []
        embeddings = []
        labels = []
        num = 0
        results = self.db.find_all()
        
        for result in results:
            print(result)
            try:
                for embedding in result['embeddings']:
                    print(num)
                    num+=1
                    e = np.fromstring(embedding[1:-1],sep=' ')
                    
                    id.append(result['employee_id'])
                    labels.append(result['name'])
                    embeddings.append(e)
            except:
                pass
        id  = np.asarray(id) 
        labels = np.asarray(labels)
        embeddings = np.array(embeddings)
        return id,embeddings,labels

    def detect_mask(self,face, size = (224,224)):
        face = cv2.resize(face,size)
        face = np.expand_dims(face,axis=0)
        result = self.mask_model.predict(face)[0]
        y_int = result.argmax()
        mask = self.convert_name(y_int)
        return mask

    def insert_person(self,data):
        return self.db.insert_person(data)

    def delete_person(self,data):
        return self.db.delete_person(data)  

    def update_person(self,data):
        return self.db.update_person(data) 
    
    def find_all(self):
        return self.db.find_all()  
    
    def get_one(self,data):
        return self.db.get_one(data)

    def insert_image(self,data):
        return self.db.insert_image(data)

    def convert_name(self,y):
        if y==0:
            return "with_mask"
        else:
            return "without_mask"   


if __name__ == "__main__":
    # arr1 = np.array([1,2,3,4,5,6])
    # arr2 = np.array([1,2,3,4,5,6])
    # arr1 = np.array_str(arr1)
    # arr2 = np.array_str(arr2)
    data = (4,'thanh',[])
   
    db = DB()
    con = Controller()
    con.insert_person(data)
    # db.insert_person(data)
    # res = db.find_all()
    # for re in res:
    #     print(re)
        # re = con.get_one(1)
        # print(re['name'])
