######## Raspberry Pi Pet Detector Camera using TensorFlow Object Detection API #########
#
# Author: Mois Brigitte-Izabella
# Date: 27/10/19
# Description:
#
# This script implements a "pet detector" that alerts the user via email if a pet is
# in the frame or not. It takes video frames from a livestream, passes them through a TensorFlow object detection model,
# determines if a cat or dog has been detected in the image, and sends an email to the user.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.mime.base import MIMEBase

IM_WIDTH = 640
IM_HEIGHT = 480

#### Initialize TensorFlow model ####

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

#### Initialize other parameters ####

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Define inside box coordinates (top left and bottom right)
#TL_bed = (int(IM_WIDTH*0.05),int(IM_HEIGHT*0.05))
#BR_bed = (int(IM_WIDTH*0.65),int(IM_HEIGHT*0.8))

# Initialize control variables used for pet detector
detected = False
frame_counter = 0
pause_counter = 0

# Initialize variables used for sending the email
smtp_server = "smtp.gmail.com"
port = 587  
sender_email = "kiwi.barksalot@gmail.com"
receiver_email= "brigittemois@yahoo.com"
password = "kiwithepup1509"
in_frame = "Hey there! Kiwi is on video, you can check at , but if you don't have time for this, I took a picture of her for you!"
out_frame= "Hey there! Kiwi is no longer on video, you can check this at ,but you don't have time for this, I took a picture for you!"
context = ssl.create_default_context()

#Method to send email with appropriate body
def send_email(body):
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "Kiwi email"
    message.attach(MIMEText(body, "plain"))
    filename = "kiwi.png"
    with open(filename, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email    
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {filename}",
    )

    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()
     # Try to log in to server and send email
    try:
        server = smtplib.SMTP(smtp_server,port)
        server.ehlo() # Can be omitted
        server.starttls(context=context) # Secure the connection
        server.ehlo() # Can be omitted
        server.login(sender_email, password)
        server.sendmail(sender_email,receiver_email,text)
    except Exception as e:
        # Print any error messages to stdout
        print(e)
    finally:
        server.quit() 

#Takes screenshot
def take_screenshot():
    cap = cv2.VideoCapture('http://192.168.1.227:8081')
    ret, image = cap.read()
    cv2.imwrite("kiwi.png", image)
    cap.release()

#### Pet detection function ####

# This function contains the code to detect a pet, determine if it's
# inside or outside, and send a text to the user's phone.
def pet_detector(frame):

    # Use globals for the control variables so they retain their value after function exits
    global detected
    global frame_counter
    global pause, pause_counter

    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})


    # Check the class of the top detected object by looking at classes[0][0].
    # If the top detected object is a cat (17) or a dog (18), find their
    if (((int(classes[0][0]) == 17) or (int(classes[0][0] == 18)))):
        frame_counter = frame_counter + 1
    else:
        pause_counter = pause_counter + 1
        
    # If pet has been detected inside for more than 5 frames, set detected_inside flag
    # and send a text to the phone.
    if ((frame_counter > 3) and (detected == False)):
        detected = True
        take_screenshot()
        send_email(in_frame)
        pause_counter = 0
        frame_counter = 0

        # Increment pause counter until it reaches 20 (for a framerate of 1.5 FPS, this is about 20 seconds),
        # then unpause the application (set pause flag to 0).
    if ((pause_counter > 20) and (detected == True)):
        pause_counter = 0
        frame_counter = 0
        detected = False
        take_screenshot()
        send_email(out_frame)
        
    # Draw counter info
    #cv2.putText(frame,'Detection counter: ' + str(frame_counter),(10,100),font,0.5,(255,255,0),1,cv2.LINE_AA)
    #cv2.putText(frame,'Pause counter: ' + str(pause_counter),(10,150),font,0.5,(255,255,0),1,cv2.LINE_AA)

    return frame

while True:
    #t1 = cv2.getTickCount()
    cap = cv2.VideoCapture('http://46.214.79.134:8080')
    ret, frame = cap.read()

    #frame = np.copy(frame1)
    #frame.setflags(write=1)

    # Pass frame into pet detection function
    frame = pet_detector(frame)

    # Draw FPS
    #cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
    
    #All the results have been drawn on the frame, so it's time to display it.
    #cv2.imshow('Object detector', frame)

    # FPS calculation
    #t2 = cv2.getTickCount()
    #time1 = (t2-t1)/freq
    #frame_rate_calc = 1/time1

    cap.release()
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
        
cv2.destroyAllWindows()

