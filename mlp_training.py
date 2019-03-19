# VII Semester Project


import cv2
import numpy as np
import glob
import sys
from sklearn.model_selection import train_test_split

print('Loading training data...')
e0 = cv2.getTickCount() #Returns number of clock-cycles after an event (FOR TIME MEASUREMENT)

# creating the arrays
image_array = np.zeros((1, 38400))
label_array = np.zeros((1, 4), 'float')
# load training data
training_data = glob.glob('training_data/*.npz')

# if no data, exit
if not training_data:
    print("No training data in directory, exit")
    sys.exit()

# loop for the npz file
for single_npz in training_data:
    with np.load(single_npz) as data:
        train_temp = data['train']
        train_labels_temp = data['train_labels']
    image_array = np.vstack((image_array, train_temp))
    label_array = np.vstack((label_array, train_labels_temp))

X = image_array[1:, :]
y = label_array[1:, :]
print('Image array shape: ', X.shape)
print('Label array shape: ', y.shape)

e00 = cv2.getTickCount() #Returns number of clock-cycles after an event (FOR TIME MEASUREMENT)
time0 = (e00 - e0)/ cv2.getTickFrequency()
print('Loading image duration:', time0)

# train test split, extracting the values fro
train, test, train_labels, test_labels = train_test_split(X, y, test_size=0.3)

# set start time
e1 = cv2.getTickCount()

# create MLP & initialize the neural network model; giving layer size
layer_sizes = np.int32([38400, 32, 4])
model = cv2.ml.ANN_MLP_create()
model.setLayerSizes(layer_sizes)
# Backpropagation algorithm
model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
model.setBackpropMomentumScale(0.0)
model.setBackpropWeightScale(0.001)
model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 20, 0.01))
# Activation function for the neural network
model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)

# Backprop parameters
criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 0.0001)

params = dict(term_crit = criteria,
               train_method = cv2.ml.ANN_MLP_BACKPROP,
               bp_dw_scale = 0.001,
               bp_moment_scale = 0.0 )

# Training function
print ('Training MLP ...')
num_iter = model.train(np.float32(train), cv2.ml.ROW_SAMPLE, np.float32(train_labels))

# set end time
e2 = cv2.getTickCount()
time = (e2 - e1)/cv2.getTickFrequency()
print ('Training duration:', time)
#print 'Ran for %d iterations' % num_iter


# saving the model in an xml file
model.save('mlp_xml/mlp.xml')

print('Ran for %d iterations' % num_iter)

# inserting training data as input and obtaining the predictions
ret, resp = model.predict(train)
prediction = resp.argmax(-1) # argmax returns maximum values in the function
print('Prediction:', prediction)
true_labels = train_labels.argmax(-1)
print('True labels:', true_labels)


#comparing the predicted values with the trained data
print('Testing...')
train_rate = np.mean(prediction == true_labels)
print('Train rate: %f' % (train_rate*100))

