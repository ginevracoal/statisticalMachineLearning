# esempio qui: http://peekaboo-vision.blogspot.com/2012/12/kernel-approximations-for-efficient.html

# import functions
import encode_csv
from functions import * 

(X_train, y_train), (X_test, y_test) = mnist.load_data()
digits = X_train

# print("train:", X_train.shape, ", test:", X_test.shape)

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]

X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# print(X_train.shape)
# normalize from 0-255 to 0-1
X_train = X_train // 255
X_test = X_test // 255

# now to -1 - 1
train_mean = X_train.mean(axis=0)
test_mean = X_test.mean(axis=0)

X_train -= train_mean
X_test -= test_mean

# not taking the whole dataset

n = 20000 # come quello del sito....
X_train = X_train[:n]
y_train = y_train[:n]
X_test = X_test[:n//20]
y_test = y_test[:n//20]

# print("train:", X_train.shape, ", test:", X_test.shape)

orig_mnist_fit = fit_all(X_train, y_train, X_test, y_test, gamma=0.031, C=100)

save(orig_mnist_fit, 'orig_mnist_script')

