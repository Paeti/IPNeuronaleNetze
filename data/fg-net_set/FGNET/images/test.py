import cv2 as cv 
import numpy as np 
import os
import json
import logging  
import requests
from PIL import Image
import base64
import tensorflow as tf
import base64

#import pickle
#import numpy as np 
#import requests  
#host = 'localhost' 
#port = '8501' batch_size = 1 
#image_path = "." 
#model_name = 'mnist' 
#signature_name = 'predict_images'  
#with open(image_path, 'rb') as f:
#    image = pickle.load(f) 
#batch = np.repeat(image, batch_size, axis=0).tolist() 
#json = { "signature_name": signature_name, "instances": batch } 
#response = requests.post("http://%s:%s/v1/models/mnist:predict" % (host, port), json=json)

#tf.enable_eager_execution()


# These two lines enable debugging at httplib level (requests->urllib3->http.client) 
# You will see the REQUEST, including HEADERS and DATA, and RESPONSE with HEADERS but without DATA. 
# The only thing missing will be the response.body which is not logged. 

try:
    import http.client as http_client 
except ImportError:     
  # Python 2     
    import httplib as http_client 
http_client.HTTPConnection.debuglevel = 1  
# You must initialize logging, otherwise you'll not see debug output. 

logging.basicConfig() 
logging.getLogger().setLevel(logging.DEBUG) 
requests_log = logging.getLogger("requests.packages.urllib3") 
requests_log.setLevel(logging.DEBUG) 
requests_log.propagate = True


test_img = cv.imread('045A32.JPG')
#print(test_img)
print(type(test_img))
resized_image = cv.resize(test_img, (224,224))
#cv.imshow('dst_rt', resized_image)
img = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)     
img = img.astype(np.float32)
print(type(img))
encoded_image_cv = base64.b64encode(img)
print(type(encoded_image_cv))


img = tf.reshape(img, [-1, 224, 224, 3])  #ggf nicht noetig Kom1

sess = tf.Session()

with sess.as_default():
    img  = img.eval() 


print("tensor to array___________________")
print(img.shape)
print(type(img))
print("___________________________________________")

#img = Image.open("045A32.JPG")          
#width, height = img.size              
#img = img.resize((224, 224))                               
#img.save("resized_picture.jpg") 
#input_image = open("resized_picture.jpg", "rb").read() 
#print("______________________________________________")  
#print(type(input_image))

#encoded_input_string = base64.b64encode(input_image) 
#input_string = encoded_input_string.decode("utf-8") 

#print(resized_image)
#print(type(resized_image))

#print(img)
#print(type(img))  

#f = open("log.txt", "w+")

#image_string = tf.read_file("045A32.JPG")         
#image_decoded = tf.image.decode_jpeg(image_string, channels=3)         
#image_float = tf.cast(image_decoded, tf.float32)         
#resize_fn = tf.image.resize_image_with_crop_or_pad         
#image_resized = resize_fn(image_float, 224, 224)         
#means = tf.reshape(tf.constant([123.68, 116.78, 103.94]), [1, 1, 3])        
#image = image_resized - means

#print(image)

instance = [{"b64": encoded_image_cv}] 
data = json.dumps({"instances": instance})

data_string = '{"instances":' + str(img.tolist()) + '}'  #ggfimg.tolist() => tensor hat kein tolist,
                                                         # s. Kom1, viell als tensor vorhanden viell nicht


data_string_json = json.dumps(data_string)

#img_list = img.tolist()
#json_format_img = json.dumps(img_list)

#payload = {}
#payload['instances'] = img_list
headers = {"content-type": "application/json"}

#f.write(json_format_img)
#print(data)

#json = {"instances":img_list}

json_response = requests.post('http://localhost:8501/v1/models/AgeVGG16:predict', data=data_string_json, headers=headers)



print("Alter:")
print(json_response)


print(img.shape)
