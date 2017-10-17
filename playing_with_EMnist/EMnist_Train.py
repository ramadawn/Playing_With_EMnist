import struct
import numpy as np
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

    

#import in dataset
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

train_img = read_idx('emnist-letters-train-images-idx3-ubyte')
train_label = read_idx('emnist-letters-train-labels-idx1-ubyte')
test_img = read_idx('emnist-letters-test-images-idx3-ubyte')
test_label = read_idx('emnist-letters-test-labels-idx1-ubyte')

#parameter

Number_Classes = 27
batch_size = 64
epochs = 20

# input image dimensions
img_rows, img_cols = 28, 28
depth = 1

#retype data to float 32

train_img = train_img.astype('float32')
test_img = test_img.astype('float32')


#Normalize image data
value_range = 255
train_img /= value_range
test_img /= value_range

#Shape Data
train_img = train_img.reshape(train_img.shape[0], img_rows, img_cols, depth)
test_img = test_img.reshape(test_img.shape[0], img_rows, img_cols, depth )
shape = (img_rows, img_cols, depth )

#One hot encoding

train_label = keras.utils.to_categorical(train_label, Number_Classes )
test_label = keras.utils.to_categorical(test_label, Number_Classes)

#initialize model

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),activation='relu',
                 input_shape=shape))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(Number_Classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(train_img, train_label,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1, 
    validation_data=(test_img,test_label))

score = model.evaluate(test_img, test_label, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#save the model
#serialize model to jason
model_json = model.to_json()
with open("EMmodel.json", "w") as json_file:
          json_file.write(model_json)

#serialize weights to HDF5
model.save_weights("EMmodel.h5")



print('ok')














##for line in range(10):
##    current_img = file[line]
##    print(current_img)
##    img = Image.fromarray(current_img)
##    img.show()
##    print("label = ", label[line])
##    input()
   


##img = Image.fromarray(data, 'RGB')
##img.save('my.png')
##img.show()

    
