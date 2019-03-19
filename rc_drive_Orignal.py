# VII Semester Project

import socket
import serial
import cv2
import time
import numpy as np
import math


class NeuralNetwork(object):
    # Class for NeuralNetwork Model
    
    def __init__(self): # creating instance of neural network model
        self.model = cv2.ml.ANN_MLP_create()
    
    def create(self):
        layer_size = np.int32([38400, 32, 4])
        self.model.setLayerSizes(layer_size)
        self.model = cv2.ml.ANN_MLP_load('mlp_xml/mlp.xml')
    
    def predict(self, samples):
        ret, resp = self.model.predict(samples)
        return resp.argmax(-1)


class RCControl(object):
    def __init__(self):
        self.serial_port = serial.Serial('COM3', 9600, timeout = 1)
    
    def steer(self, prediction):
        if prediction == 2:
            self.serial_port.write("8".encode())
            print("Forward")
        elif prediction == 0:
            self.serial_port.write("4".encode())
            print("Left")
        elif prediction == 1:
            self.serial_port.write("6".encode())
            print("Right")
        else:
            self.stop()
    
    def stop(self):
        self.serial_port.write("0".encode())

class Execute(object):

    def __init__(self):
        self.socket = socket.socket()
        self.connection = self.socket.connect(('192.168.0.6', 8081))
        self.MainProgram()

    def MainProgram(self):
        NeuralModel = NeuralNetwork()
        NeuralModel.create()
        rc_car = RCControl()
        delay = 0
        stream_bytes = b' '
        # stream video frames one by one
        try:
            while True:
                stream_bytes += self.socket.recv(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    gray = cv2.imdecode(np.fromstring(jpg, dtype = np.uint8), cv2.IMREAD_GRAYSCALE)
                    image = cv2.imdecode(np.fromstring(jpg, dtype = np.uint8), cv2.IMREAD_UNCHANGED)
                    # lower half of the image
                    roi = gray[120:240, :]
                    cv2.imshow('image', image)
                    # cv2.imshow('mlp_image', half_gray)
                    # reshape image
                    image_array = roi.reshape(1, 38400).astype(np.float32)
                    # neural network makes prediction
                    # prediction = self.model.predict(image_array)
                    prediction = NeuralModel.predict(image_array)
                    if(delay == 0):
                        rc_car.steer(prediction)
                        delay = 30
                    delay = delay - 1
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    #time.sleep(1)
        
            cv2.destroyAllWindows()
        finally:
            print("Connection closed on thread 1")
        

if __name__ == '__main__':
    Execute()

    