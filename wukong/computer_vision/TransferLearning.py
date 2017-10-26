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


def default_pretrained_model( img_width, img_height ):
    return VGG16(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))


def default_top_model( input_dim ):
    model = Sequential()
    model.add(Flatten(input_shape=input_dim))
    model.add(Dropout(0.7))
    model.add(Dense(1, activation='sigmoid'))
    return model


def extract_feature( model, target_width, target_height, img_file ):
    img = kimage.load_img(img_file, target_size=(target_width, target_height))
    x = kimage.img_to_array(img)
    x = np.expand_dims(x / 255., axis=0)
    return model.predict(x)[0]


def extract_features_of_samples( model, target_width, target_height, data_dir, features_file_name ):
    category_id = 0
    features = []
    feature_labels = []
    for file_name in os.listdir(data_dir):
        fullname = os.path.join(data_dir, file_name)
        if os.path.isdir(fullname):
            for img_file in os.listdir(fullname):
                img_file_fullname = os.path.join(fullname, img_file)
                print (img_file_fullname)
                feature = extract_feature(model, target_width, target_height, img_file_fullname)
                features.append(feature)
                feature_labels.append(category_id)
            category_id = category_id + 1
            print("ID:{%d} represents {%s}" % (category_id, file_name))
    np.save(open(features_file_name, 'w'),
            np.array(features))
    np.save(open(features_file_name + ".label", 'w'),
            np.array(feature_labels))


def train_top_model( top_model, bottlebeck_features_file_name, top_model_weights_path, batch_size, epochs ):
    train_data = np.load(open(bottlebeck_features_file_name + '.train'))
    num_of_training_samples = train_data.shape[0]
    print "train on {%d} samples" % num_of_training_samples
    train_labels = np.load(open(bottlebeck_features_file_name + ".train.label"))

    validation_data = np.load(open(bottlebeck_features_file_name + '.test'))
    num_of_validation_samples = validation_data.shape[0]
    print "validate on {%d} samples" % num_of_validation_samples

    validation_labels = np.load(open(bottlebeck_features_file_name + ".test.label"))
    top_model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      loss='binary_crossentropy', metrics=['accuracy'])

    monitor = EarlyStopping(monitor='loss', min_delta=0.0001, patience=5, verbose=0, mode='auto')
    check_filepath = top_model_weights_path + ".best.hdf5"
    checkpoint = ModelCheckpoint(check_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    history = History()
    top_model.fit(train_data, train_labels,
                  nb_epoch=epochs,
                  batch_size=batch_size,
                  validation_data=(validation_data, validation_labels),
                  callbacks=[checkpoint, monitor, history])
    top_model.load_weights(check_filepath)
    return (num_of_training_samples, num_of_validation_samples)


def refactor_pretrained_model( pretrained_model, top_model, num_of_freezed_layers ):
    for layer in pretrained_model.layers[:num_of_freezed_layers]:
        layer.trainable = False
    combined_model = Model(pretrained_model.input,
                           top_model(pretrained_model.output))
    return combined_model


def fine_tune( combined_model, img_height, img_width, epochs, batch_size,
               train_data_dir, nb_train_samples,
               validation_data_dir, nb_validation_samples,
               combined_model_weights_file ):
    combined_model.compile(loss='binary_crossentropy',
                           optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                           metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.4,
        zoom_range=0.4,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False,
        class_mode='binary')

    monitor = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=6, verbose=0, mode='auto')
    check_filepath = combined_model_weights_file + "-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(check_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    print ("nb_val_samples", nb_validation_samples)
    combined_model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=epochs,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples,
        callbacks=[checkpoint, monitor]
        # validation_steps=nb_validation_samples // batch_size
    )
    combined_model.save(combined_model_weights_file)


def create_deep_vision_model( work_dir, model_name, pretrained_model, top_model, img_width, img_height,
                              train_data_dir,
                              validation_data_dir,
                              top_training_batch_size, top_training_epochs,
                              num_of_freezed_layers, fine_tuning_batch_size, fine_tuning_epochs,
                              ):
    bottlebeck_features_file_name = os.path.join(work_dir, model_name + ".bottlebeck_features")
    top_model_weights_file_name = os.path.join(work_dir, model_name + ".top_weights")
    print os.path.exists(bottlebeck_features_file_name)
    if not (os.path.exists(bottlebeck_features_file_name + ".test.label")):
        extract_features_of_samples(pretrained_model, img_width, img_height, train_data_dir,
                                    bottlebeck_features_file_name + ".train")
        extract_features_of_samples(pretrained_model, img_width, img_height, validation_data_dir,
                                    bottlebeck_features_file_name + ".test")

    print "Top layer training samples have been created!"
    num_of_training_samples, num_of_validation_samples = train_top_model(top_model, bottlebeck_features_file_name,
                                                                         top_model_weights_file_name,
                                                                         top_training_batch_size, top_training_epochs)
    print "Top layer training was done!"
    combined_model = refactor_pretrained_model(pretrained_model, top_model,
                                               num_of_freezed_layers)
    combined_model_weights_file_name = work_dir + "/" + model_name + ".combined_model_weights"
    print os.path.exists(combined_model_weights_file_name)
    if os.path.exists(combined_model_weights_file_name):
        combined_model.load_weights(combined_model_weights_file_name)
    fine_tune(combined_model, img_width, img_height, fine_tuning_epochs, fine_tuning_batch_size,
              train_data_dir, num_of_training_samples,
              validation_data_dir, num_of_validation_samples,
              combined_model_weights_file_name)
    print ("Done! The weights file is " + combined_model_weights_file_name)


def create_default_deep_vision_model( work_dir, model_name,
                                      train_data_dir,
                                      validation_data_dir,
                                      top_training_batch_size=16, top_training_epochs=50,
                                      num_of_freezed_layers=15, fine_tuning_batch_size=16, fine_tuning_epochs=50,
                                      img_width=400, img_height=400
                                      ):
    pretrained_model = default_pretrained_model(img_width, img_height)
    top_model = default_top_model(pretrained_model.output_shape[1:])
    create_deep_vision_model(work_dir, model_name, pretrained_model, top_model, img_width, img_height,
                             train_data_dir,
                             validation_data_dir,
                             top_training_batch_size, top_training_epochs,
                             num_of_freezed_layers, fine_tuning_batch_size, fine_tuning_epochs,
                             )
