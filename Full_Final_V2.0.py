import cv2
import sys
# import RPi.GPIO as GPIO
import time
import matplotlib.pyplot as mat
from math import atan2, sqrt, cos, sin, pi
import math
import numpy as np

#
# def encoder_handler(A):
#     global counter_right
#     global counter_left
#     global prev_counter_right
#     global prev_counter_left
#
#     global delta_ticks
#     # if A == "r":
#     #     prev_counter_right = counter_right
#     #     counter_right += 1
#     # if A == "l":
#     #     prev_counter_left =  counter_left
#     #     counter_left += 1
#
#     if A == "r":
#         prev_counter_right = counter_right
#         if GPIO.input(m1_enB) == GPIO.input(m1_enA):
#             counter_right += 1
#         elif GPIO.input(m1_enB) != GPIO.input(m1_enA):
#             counter_right -= 1
#     if A == "l":
#         prev_counter_left = counter_left
#
#         if GPIO.input(m1_enB) == GPIO.input(m1_enA):
#             counter_left += 1
#         elif GPIO.input(m1_enB) != GPIO.input(m1_enA):
#             counter_left -= 1
#
#
# def setMotor(dire, pwmVal, in1, in2, motor):
#     if dire == 1:
#         GPIO.output(in1, 1)
#         GPIO.output(in2, 0)
#     elif dire == -1:
#         GPIO.output(in1, 0)
#         GPIO.output(in2, 1)
#     else:
#         GPIO.output(in1, 0)
#         GPIO.output(in2, 0)
#     my_pwm2.start(pwmVal)
#     if motor == 2:
#         my_pwm2.start(pwmVal)
#     elif motor == 1:
#         my_pwm1.start(pwmVal)
#
#
# def setting_up_GPIO():
#     GPIO.cleanup()
#     global m1_enA
#     m1_enA = 11
#     global m1_enB
#     m1_enB = 12
#     global m2_enA
#     m2_enA = 13
#     global m2_enB
#     m2_enB = 15
#     global m2_input1
#     m2_input1 = 23
#     global m2_input2
#     m2_input2 = 19
#     global m1_input1
#     m1_input1 = 22
#     global m1_input2
#     m1_input2 = 21
#     pwm2 = 33
#     pwm1 = 32
#
#     GPIO.setmode(GPIO.BOARD)
#     GPIO.setup(m1_enA, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
#     GPIO.setup(m2_enA, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
#     GPIO.setup(m1_enB, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
#     GPIO.setup(m2_enB, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
#     GPIO.setup(pwm2, GPIO.OUT)
#     GPIO.setup(m2_input1, GPIO.OUT)
#     GPIO.setup(m2_input2, GPIO.OUT)
#     GPIO.setup(pwm1, GPIO.OUT)
#     GPIO.setup(m1_input1, GPIO.OUT)
#     GPIO.setup(m1_input2, GPIO.OUT)
#     global my_pwm2
#     my_pwm2 = GPIO.PWM(pwm2, 100)
#     global my_pwm1
#     my_pwm1 = GPIO.PWM(pwm1, 100)
#
#
# def stop():
#     my_pwm1.ChangeDutyCycle(0)
#     my_pwm2.ChangeDutyCycle(0)
#
#
# setting_up_GPIO()
source = cv2.VideoCapture(0)

win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
net = cv2.dnn.readNetFromCaffe("deploy.prototxt",
                               "res10_300x300_ssd_iter_140000_fp16.caffemodel")
in_width = 300
in_height = 300
# FOR THE ORIGINAL FULL WINDOW reg_height, reg_width = (0, 0), (639, 475)
x_left_original = 150
y_left_original = 100
x_right_original = 489
y_right_original = 375

x_left_bottom = 200
y_left_bottom = 150
x_right_top = 400
y_right_top = 300
command = ""
prev_command = ""
reg_height, reg_width = (150, 100), (489, 375)
# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2

# here is the list of the mean values from each of the color channels across all the images that were used in training
mean = [104, 117, 123]
# confidence threshold is a value we can set that will determine the sensitivity of our detections
conf_threshold = 0.7

while cv2.waitKey(1) != 27:
    has_frame, frame = source.read()
    if not has_frame:
        break
    # flipping the photoage that we are using
    frame = cv2.flip(frame, 1)
    # retrieving the size the frame
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    cv2.circle(frame, (150, 100), 15, (0, 255, 0), -1)
    cv2.line(frame, (150, 0), (150, 1200), (0, 255, 0), 4)
    cv2.line(frame, (0, 100), (1600, 100), (0, 255, 0), 4)
    cv2.circle(frame, (489, 375), 15, (0, 0, 255), -1)
    cv2.line(frame, (489, 0), (489, 1200), (0, 0, 255), 4)
    cv2.line(frame, (0, 375), (1600, 375), (0, 0, 255), 4)


    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
    # Run a model
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
            y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
            x_right_top = int(detections[0, 0, i, 5] * frame_width)
            y_right_top = int(detections[0, 0, i, 6] * frame_height)

            cv2.circle(frame, (x_left_bottom, y_left_bottom), 15, (0, 255, 0), -1)
            cv2.circle(frame, (x_right_top, y_right_top), 15, (0, 0, 255), -1)

            cv2.circle(frame, (x_left_bottom, y_left_bottom), 15, (0, 255, 0), -1)
            cv2.line(frame, (x_left_bottom, 0), (x_left_bottom, 1200), (0, 255, 0), 4)
            cv2.line(frame, (0, y_left_bottom), (1600, y_left_bottom), (0, 255, 0), 4)
            cv2.circle(frame, (x_right_top, y_right_top), 15, (0, 0, 255), -1)
            cv2.line(frame, (x_right_top, 0), (x_right_top, 1200), (0, 0, 255), 4)
            cv2.line(frame, (0, y_right_top), (1600, y_right_top), (0, 0, 255), 4)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))
            label = "Confidence: %.4f" % confidence
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom - label_size[1]),
                          (x_left_bottom + label_size[0], y_left_bottom + base_line),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (x_left_bottom, y_left_bottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    """
    this about calculating the command that we will give to the arduino
    """
    if x_left_bottom <= x_left_original:
        command = 'move_left'  # right
        prev_command = command
        # setMotor(1, 35, m1_input1, m1_input2, 1)
        # setMotor(-1, 35, m2_input1, m2_input2, 2)
    elif x_right_top >= x_right_original:
        command = 'move_right'  # left
        prev_command = command
        # setMotor(-1, 35, m1_input1, m1_input2, 1)
        # setMotor(1, 35, m2_input1, m2_input2, 2)

    elif y_left_bottom <= y_left_original:
        command = 'move_forward'  # backward
        prev_command = command
        # setMotor(1, 60, m1_input1, m1_input2, 1)
        # setMotor(1, 60, m2_input1, m2_input2, 2)
    elif y_right_top >= y_right_original:
        command = 'move_backward'  # forward
        prev_command = command
        # setMotor(-1, 60, m1_input1, m1_input2, 1)
        # setMotor(-1, 60, m2_input1, m2_input2, 2)
    elif x_left_bottom <= x_left_original and x_right_top >= x_right_original and y_right_top >= y_right_original and y_left_bottom <= y_left_original:
        command = 'move_backward'  # forward
        prev_command = command
        # setMotor(-1, 60, m1_input1, m1_input2, 1)
        # setMotor(-1, 60, m2_input1, m2_input2, 2)
    else:
        command = 'stop'  # stop
        prev_command = command
        # setMotor(-1, 0, m1_input1, m1_input2, 1)
        # setMotor(-1, 0, m2_input1, m2_input2, 2)
        # stop()
    # if command != prev_command:frame.shape[0]
    #     serialInit.write(command.encodeframe.shape[0]('utf-8'))
    # serialInit.write(command.encode('utframe.shape[0]f-8'))

    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency()) + "------ Command: " + command
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    # frame = cv2.rectangle(frame, reg_height,reg_width, color, thickness)

    cv2.imshow(win_name, frame)

# serialInit.close()
# stop()
source.release()
cv2.destroyWindow(win_name)