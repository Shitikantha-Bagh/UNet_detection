#Video Testing of UNet
#Fragments of the notebook, TO DO: update complete code

import cv2
import tensorflow as tf
from tensorflow import _keras
from keras.models import load_model
import numpy as np


model = load_model('NARROWJOINTmodel_unet.keras', compile=False)



def preprocess_image(image_path,target_size):
    
    capture = cv2.cvtColor(image_path,cv2.COLOR_BGR2RGB)
    capture = cv2.resize(capture,target_size)
    capture = capture/255
    capture = np.expand_dims(capture, axis=0)
    return capture

def predict_image(model,image):
    prediction_video = model.predict(image)
    return prediction_video

cap = cv2.VideoCapture("/dev/v4l/by-id/usb-046d_Logitech_BRIO_6AB57283-video-index0")




while True:
    ret,frame = cap.read()
    #print(frame)
   
    if not ret:
        break


    input_image = preprocess_image(frame,target_size=(256,256))
    #print(input_image)


    prediction_v = predict_image(model,input_image)

    mask = (prediction_v[0,:,:,0]>0.5).astype(np.uint8)*255
    #print(mask.shape)
    #print(mask.dtype)
 

    #cv2.imshow('Original frame',input_image)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()