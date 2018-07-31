import os

import numpy as np
from keras import optimizers
from keras.applications import VGG16
from keras.callbacks import EarlyStopping
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing import image as kimage
from keras.preprocessing.image import ImageDataGenerator

'''
Author: Cai Chao
Created Date: 2017-10-27
Contact me: chaocai2001@icloud.com


'''


class WuKongVisionModel:
    """
        Encapsulate the transfer learning training process.
        Given top model and pretrained model, the class would train a new combined model for new task
        with the typical transfer learning process.
    """

    def __init__( self, target_width=224, target_height=224, pretrained_model=None, top_model=None,
                  num_of_frozen_layers=0 ):
        self._target_width = target_width
        self._target_height = target_height
        self._top_model = top_model
        self._num_of_frozen_layers = num_of_frozen_layers
        if (pretrained_model is None):
            self._pretrained_model = self._default_pretrained_model()
        else:
            self._pretrained_model = pretrained_model
        if (top_model is None):
            self._top_model = self._default_top_model(self._pretrained_model.output_shape[1:])
        else:
            self._top_model = top_model

    def load_weights( self, weights_file ):
        self._refactor_pretrained_model()
        self._combined_model.load_weights(weights_file)

    def predict( self, img_file ):
        img = self._preprocess_image_file(img_file)
        return self._combined_model.predict(img)[0]

    def _default_pretrained_model( self ):
        self._num_of_frozen_layers = 15
        return VGG16(include_top=False, weights='imagenet',
                     input_shape=(self._target_width, self._target_height, 3))

    def _default_top_model( self, input_dim ):
        model = Sequential()
        model.add(Flatten(input_shape=input_dim))
        model.add(Dropout(0.7))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def _refactor_pretrained_model( self ):
        for layer in self._pretrained_model.layers[:self._num_of_frozen_layers]:
            layer.trainable = False
        self._combined_model = Model(self._pretrained_model.input,
                                         self._top_model(self._pretrained_model.output))

    def _preprocess_image_file( self, img_file ):
        img = kimage.load_img(img_file, target_size=(self._target_width, self._target_height))
        x = kimage.img_to_array(img)
        x = np.expand_dims(x / 255., axis=0)
        return x

    def _extract_feature( self, img_file ):
        return self._pretrained_model.predict(self._preprocess_image_file(img_file))[0]

    def _extract_features_of_samples( self, data_dir, features_file_name ):
        category_id = 0
        features = []
        feature_labels = []
        for file_name in os.listdir(data_dir):
            fullname = os.path.join(data_dir, file_name)
            if os.path.isdir(fullname):
                for img_file in os.listdir(fullname):
                    img_file_fullname = os.path.join(fullname, img_file)
                    feature = self._extract_feature(img_file_fullname)
                    features.append(feature)
                    feature_labels.append(category_id)
                print("ID:%d represents %s" % (category_id, file_name))
                category_id = category_id + 1
        np.save(open(features_file_name, 'w'),
                np.array(features))
        np.save(open(features_file_name + ".label", 'w'),
                np.array(feature_labels))

    def _train_top_model( self, bottlebeck_features_file_name, top_model_weights_path, batch_size, epochs ):
        train_data = np.load(open(bottlebeck_features_file_name + '.train'))
        num_of_training_samples = train_data.shape[0]
        print "train on {%d} samples" % num_of_training_samples
        train_labels = np.load(open(bottlebeck_features_file_name + ".train.label"))

        validation_data = np.load(open(bottlebeck_features_file_name + '.test'))
        num_of_validation_samples = validation_data.shape[0]
        print "validate on {%d} samples" % num_of_validation_samples

        validation_labels = np.load(open(bottlebeck_features_file_name + ".test.label"))
        self._top_model.compile(optimizer="rmsprop",
                                loss='binary_crossentropy', metrics=['accuracy'])

        monitor = EarlyStopping(monitor='acc', min_delta=0.001, patience=5, verbose=0, mode='max')
        check_filepath = top_model_weights_path + ".best.hdf5"
        checkpoint = ModelCheckpoint(check_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        history = History()
        self._top_model.fit(train_data, train_labels,
                            nb_epoch=epochs,
                            batch_size=batch_size,
                            validation_data=(validation_data, validation_labels),
                            callbacks=[checkpoint, monitor, history])
        self._top_model.load_weights(check_filepath)
        self._top_model.compile(optimizer="rmsprop",
                                loss='binary_crossentropy', metrics=['accuracy'])
        self._num_of_training_samples = num_of_training_samples
        self._num_of_validation_samples = num_of_validation_samples

    def _fine_tune( self, epochs, batch_size,
                    train_data_dir, validation_data_dir,
                    combined_model_weights_file, class_mode, loss, optimizer ):
        self._combined_model.compile(loss=loss,
                                     optimizer=optimizer,
                                     metrics=['accuracy'])

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.4,
            zoom_range=0.4,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(self._target_height, self._target_width),
            batch_size=batch_size,
            class_mode=class_mode)

        validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(self._target_height, self._target_width),
            batch_size=batch_size,
            class_mode=class_mode)

        monitor = EarlyStopping(monitor='acc', min_delta=0.001, patience=6, verbose=0, mode='max')
        check_filepath = combined_model_weights_file + ".best.hdf5"
        checkpoint = ModelCheckpoint(check_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        self._combined_model.fit_generator(
            train_generator,
            steps_per_epoch = self._num_of_training_samples // batch_size,
            #samples_per_epoch=self._num_of_training_samples,
            epochs=epochs,
            validation_data=validation_generator,
            #nb_val_samples=self._num_of_validation_samples,
            callbacks=[checkpoint, monitor],
            validation_steps=self._num_of_validation_samples // batch_size
        )

    def train_for_new_task( self, work_dir, task_name,
                            train_data_dir,
                            validation_data_dir,
                            top_training_batch_size=16, top_training_epochs=50,
                            fine_tuning_batch_size=16, fine_tuning_epochs=50,
                            class_mode='binary', loss='binary_crossentropy',
                            optimizer=optimizers.SGD(lr=3e-5, momentum=0.9)
                            ):

        bottlebeck_features_file_name = os.path.join(work_dir, task_name + ".bottlebeck_features")
        top_model_weights_file_name = os.path.join(work_dir, task_name + ".top_weights")
        if not (os.path.exists(bottlebeck_features_file_name + ".test.label")):
            print ("Preparing the data for training top layer ...")
            self._extract_features_of_samples(train_data_dir,
                                              bottlebeck_features_file_name + ".train")
            self._extract_features_of_samples(validation_data_dir,
                                              bottlebeck_features_file_name + ".test")

        print "Top layer training data have been created!"
        self._train_top_model(bottlebeck_features_file_name, top_model_weights_file_name,
                              top_training_batch_size, top_training_epochs)
        print "Top layer training was done!"
        self._refactor_pretrained_model()
        combined_model_weights_file_name = work_dir + "/" + task_name + ".combined_model_weights"

        self._fine_tune(epochs=fine_tuning_epochs, batch_size=fine_tuning_batch_size,
                        train_data_dir=train_data_dir, validation_data_dir=validation_data_dir,
                        combined_model_weights_file=combined_model_weights_file_name,
                        class_mode=class_mode, loss=loss, optimizer=optimizer)
        print ("Fine tuning is done!")
