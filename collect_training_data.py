# VII Semester Project

import numpy as np
import cv2
import serial
import pygame
from pygame.locals import *
import socket
import time
import os


class CollectTrainingData(object):
    def __init__(self):
        
        # Opens up a connection @ IP : port
        self.socket = socket.socket()
        self.connection = self.socket.connect(('192.168.0.6', 8081))
        # self.streaming()
        
        # connect to a seral port @ COM3
        self.ser = serial.Serial('COM3', 9600, timeout = 1)
        self.send_inst = True
        
        # create labels
        self.k = np.zeros((4, 4), 'float')
        for i in range(4):
            self.k[i, i] = 1
        self.temp_label = np.zeros((1, 4), 'float')
        
        #Opens pygame console to control RC Car
        pygame.init()
        screen = pygame.display.set_mode((400, 300))
        self.collect_image()
    
    def collect_image(self):
        
        saved_frame = 0
        total_frame = 0
        
        # collect images for training
        print('Start collecting images...')
        e1 = cv2.getTickCount() #Returns number of clock-cycles after an event (FOR TIME MEASUREMENT)
        image_array = np.zeros((1, 38400)) # 320 x 120  = 38,400 ; QVGA = 320 X 240
        label_array = np.zeros((1, 4), 'float') # 4 instructions; forward,left,right,reverse
        
        # stream video frames one by one
        try:
            stream_bytes = b' '
            frame = 1
            while self.send_inst:
                stream_bytes += self.socket.recv(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.fromstring(jpg, dtype = np.uint8), cv2.IMREAD_GRAYSCALE)
                    
                    # region of interest is lower half of the image
                    roi = image[120:240, :]
                    
                    # save streamed images
                    cv2.imwrite('training_images/frame{:>05}.jpg'.format(frame), image)
                    
                    # displays the stream
                    cv2.imshow('image', image)
                    
                    # reshape the roi image into one row array
                    temp_array = roi.reshape(1, 38400).astype(np.float32)
                    
                    frame += 1
                    total_frame += 1
                    
                    # get input from human driver
                    for event in pygame.event.get():
                        if event.type == KEYDOWN:
                            key_input = pygame.key.get_pressed()
                            
                            # Arduino program was controlling car with 8,6,4 here we shift control to arrow keys
                            
                            # Changes input method from 8,4,6 to arrow keys; saves frame, updates image and label array
                            if key_input[pygame.K_RIGHT]:
                                print("Forward Right")
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[1]))
                                saved_frame += 1
                                self.ser.write("6".encode())
                            
                            elif key_input[pygame.K_LEFT]:
                                print("Forward Left")
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[0]))
                                saved_frame += 1
                                self.ser.write("4".encode())
                            
                            
                            # simple orders
                            elif key_input[pygame.K_UP]:
                                print("Forward")
                                saved_frame += 1
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[2]))
                                self.ser.write("8".encode())
                            
                            elif key_input[pygame.K_DOWN]:
                                print("Reverse")
                                saved_frame += 1
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[3]))
                                self.ser.write("2".encode())
                            
                            
                            elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                                print('exit')
                                self.send_inst = False
                                self.ser.write("0".encode())
                                break
                        
                        elif event.type == pygame.KEYUP:
                            self.ser.write("0".encode())
            
            # save training images and labels
            train = image_array[1:, :]
            train_labels = label_array[1:, :]
            
            # save training data as a numpy file
            file_name = str(int(time.time()))
            directory = "training_data"
            if not os.path.exists(directory):
                os.makedirs(directory)
            try:
                np.savez(directory + '/' + file_name + '.npz', train = train, train_labels = train_labels)
            except IOError as e: # Exception runs if the directory or the file wasn't found
                print(e)
            
            e2 = cv2.getTickCount()
            # calculate streaming duration
            time0 = (e2 - e1) / cv2.getTickFrequency() # Returns frequency/number of clock-cycles per second; time0 gives the time
            print('Streaming duration:', time0)
            
            print(train.shape)
            print(train_labels.shape)
            print('Total frame:', total_frame)
            print('Saved frame:', saved_frame)
            print('Dropped frame', total_frame - saved_frame)
        
        finally:
            self.connection.close()
            self.socket.close()


if __name__ == '__main__':
    CollectTrainingData()