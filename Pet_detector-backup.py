######## Raspberry Pi Pet Detector Camera using TensorFlow Object Detection API #########
#
# Author: Evan Juras
# Date: 10/15/18
# Description:
#
# This script implements a "pet detector" that alerts the user if a pet is
# waiting to be let inside or outside. It takes video frames from a Picamera
# or USB webcam, passes them through a TensorFlow object detection model,
# determines if a cat or dog has been detected in the image, checks the location
# of the cat or dog in the frame, and texts the user's phone if a cat or dog is
# detected in the appropriate location.
#
# The framework is based off the Object_detection_picamera.py script located here:
# https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/blob/master/Object_detection_picamera.py
#
# Sending a text requires setting up a Twilio account (free trials are available).
# Here is a good tutorial for using Twilio:
# https://www.twilio.com/docs/sms/quickstart/python


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
TL_bed = (int(IM_WIDTH*0.05),int(IM_HEIGHT*0.05))
BR_bed = (int(IM_WIDTH*0.65),int(IM_HEIGHT*0.8))

# Initialize control variables used for pet detector
detected_bed = False

bed_counter = 0

pause = 0
pause_counter = 0

smtp_server = "smtp.gmail.com"
port = 587  # For starttls
sender_email = "kiwi.barksalot@gmail.com"
receiver_email= "brigittemois@yahoo.com"
password = "kiwithepup1509"
body_on_bed = "Hey there! Kiwi is on the bed, safe and sound!"
body_off_bed = "Hey there! Kiwi is off the bed, who knows where!"
context = ssl.create_default_context()

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
    global detected_bed
    global bed_counter
    global pause_counter

    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
#     vis_util.visualize_boxes_and_labels_on_image_array(
#         frame,
#         np.squeeze(boxes),
#         np.squeeze(classes).astype(np.int32),
#         np.squeeze(scores),
#         category_index,
#         use_normalized_coordinates=True,
#         line_thickness=8,
#         min_score_thresh=0.40)

    # Draw boxes defining "outside" and "inside" locations.
    #cv2.rectangle(frame,TL_bed,BR_bed,(255,20,20),3)
    #cv2.putText(frame,"On bed",(TL_bed[0]+10,TL_bed[1]-10),font,1,(255,20,255),3,cv2.LINE_AA)
    
    # Check the class of the top detected object by looking at classes[0][0].
    # If the top detected object is a cat (17) or a dog (18) (or a teddy bear (88) for test purposes),
    # find its center coordinates by looking at the boxes[0][0] variable.
    # boxes[0][0] variable holds coordinates of detected objects as (ymin, xmin, ymax, xmax)
    if (((int(classes[0][0]) == 17) or (int(classes[0][0] == 18))) and (pause == 0)):
        x = int(((boxes[0][0][1]+boxes[0][0][3])/2)*IM_WIDTH)
        y = int(((boxes[0][0][0]+boxes[0][0][2])/2)*IM_HEIGHT)
    else:
        x = 0
        y = 0

        # Draw a circle at center of object
        cv2.circle(frame,(x,y), 5, (75,13,180), -1)

        # If object is in inside box, increment inside counter variable
        if ((x > TL_bed[0]) and (x < BR_bed[0]) and (y > TL_bed[1]) and (y < BR_bed[1])):
            bed_counter = bed_counter + 1
        else:
            pause_counter = pause_counter + 1

    # If pet has been detected inside for more than 5 frames, set detected_inside flag
    # and send a text to the phone.
    if ((bed_counter > 3) and (detected_bed == False)):
        detected_bed = True
        take_screenshot()
        send_email(body_on_bed)
        pause_counter = 0
        bed_counter = 0
        # Pause pet detection by setting "pause" flag
        #pause = 1

    # If pause flag is set, draw message on screen.
    #if pause == 1:
        #if detected_bed == True:
             #cv2.putText(frame,'Kiwi is on bed!',(int(IM_WIDTH*.1),int(IM_HEIGHT*.5)),font,3,(0,0,0),7,cv2.LINE_AA)
             #cv2.putText(frame,'Kiwi is on bed!',(int(IM_WIDTH*.1),int(IM_HEIGHT*.5)),font,3,(95,176,23),5,cv2.LINE_AA)

        # Increment pause counter until it reaches 30 (for a framerate of 1.5 FPS, this is about 20 seconds),
        # then unpause the application (set pause flag to 0).
    if ((pause_counter > 30) and (detected_bed == True)):
            #pause = 0
        pause_counter = 0
        bed_counter = 0
        detected_bed = False
        take_screenshot()
        send_email(body_off_bed)
        
    # Draw counter info
    cv2.putText(frame,'Detection counter: ' + str(bed_counter),(10,100),font,0.5,(255,255,0),1,cv2.LINE_AA)
    cv2.putText(frame,'Pause counter: ' + str(pause_counter),(10,150),font,0.5,(255,255,0),1,cv2.LINE_AA)

    return frame

while True:
    cap = cv2.VideoCapture('http://192.168.1.227:8081')
    ret, frame1 = cap.read()

    frame = np.copy(frame1)
    frame.setflags(write=1)

    # Pass frame into pet detection function
    frame = pet_detector(frame)

    # Draw FPS
    #cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
    
    #All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # FPS calculation
    #t2 = cv2.getTickCount()
    #time1 = (t2-t1)/freq
    #frame_rate_calc = 1/time1

    cap.release()
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
        
cv2.destroyAllWindows()
