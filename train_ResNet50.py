from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import tensorflow as tf 
import keras.backend.tensorflow_backend as KTF 
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Input
#from keras_resnet import ResNet50
from keras_resnet.models._2d import ResNet2D18

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)

epochs = 10
batch_size = 32
train_data_dir = "../dataset/driver_dataset/train/"
val_data_dir = "../dataset/driver_dataset/val/"
model_path = "./model/weights.best.resnet50.hdf5"

def setup_data(train_data_dir, batch_size):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2
                                       )
    
    train_generator = train_datagen.flow_from_directory(
        directory=train_data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=5
    )

    val_generator = train_datagen.flow_from_directory(
        directory=val_data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical"
    )

    return train_generator, val_generator

def build_model():
    base_model = ResNet50(input_shape=(224,224,3), weights=None, include_top=False)
    X = base_model.output
    predictions = Dense(10, activation='softmax')(X)
    model = Model(inputs=base_model.input, outputs=predictions)
    #model = ResNet50(Input((224, 224, 3)), classes=10)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
    return model

#def build_model():
#    inputs = Input(shape=(224,224,3), name="input")
#    
#    #Convolution 1
#    conv1 = Conv2D(128, kernel_size=(3,3), activation="relu", name="conv_1")(inputs)
#    pool1 = MaxPooling2D(pool_size=(2, 2), name="pool_1")(conv1)
#
#    #Convolution 2
#    conv2 = Conv2D(64, kernel_size=(3,3), activation="relu", name="conv_2")(pool1)
#    pool2 = MaxPooling2D(pool_size=(2, 2), name="pool_2")(conv2)
#    
#    #Convolution 3
#    conv3 = Conv2D(32, kernel_size=(3,3), activation="relu", name="conv_3")(pool2)
#    pool3 = MaxPooling2D(pool_size=(2, 2), name="pool_3")(conv3)
#    
#    #Convolution 4
#    conv4 = Conv2D(16, kernel_size=(3,3), activation="relu", name="conv_4")(pool3)
#    pool4 = MaxPooling2D(pool_size=(2, 2), name="pool_4")(conv4)
#    
#    #Fully Connected Layer
#    flatten = Flatten()(pool4)
#    fc1 = Dense(1024, activation="relu", name="fc_1")(flatten)
#    
#    #output
#    output=Dense(10, activation="softmax", name ="softmax")(fc1)
#    
#    # finalize and compile
#    model = Model(inputs=inputs, outputs=output)
#    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
#    return model

def fit_model(model, train_generator, val_generator, epoches, model_path):
    cp = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='max')
    
    tb = TensorBoard(log_dir='./log_ResNet50',
                     histogram_freq=1,
                     batch_size=32,
                     write_graph=True,
                     write_grads=True,
                     write_images=True
                     )

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        verbose=1,
        callbacks=[tb,cp]
        )
    return model

def eval_model(model, val_generator, batch_size):
    scores = model.evaluate_generator(val_generator, steps=val_generator.samples // batch_size)
    print("Loss: " + str(scores[0]) + " Accuracy: " + str(scores[1]))
    print(scores)

train_generator, val_generator = setup_data(train_data_dir, batch_size)
model = build_model()
model = fit_model(model, train_generator, val_generator, epochs, model_path)

eval_model(model, val_generator, batch_size=batch_size)







