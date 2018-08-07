import os
from keras import optimizers
from wukong.computer_vision.TransferLearning import WuKongVisionModel

train_data_dir = r'../../samples/cat_dog/train/'
test_data_dir = r'../../samples/cat_dog/test'
work_dir = r'../../tmp'
task_name = "cat_dog"

# train a model with the default configuration
model = WuKongVisionModel()
model.train_for_new_task(work_dir, task_name, train_data_dir, test_data_dir)

# predict by the trained model
#ret = model.predict(os.path.join(test_data_dir, "cat", "cat.983.jpg"))
#print ret

# Improvement tuning
#org_model = WuKongVisionModel()
#org_model.load_weights(os.path.join(work_dir, 'cat_dog.combined_model_weightsacc0.84_val_acc0.94.best.hdf5'))
#org_model.train_for_improvement_task(work_dir, task_name, train_data_dir, test_data_dir,
#                             optimizer=optimizers.SGD(lr=1e-5, momentum=0.9))
# You can load the weights to the model
#new_model = WuKongVisionModel()
#new_model.load_weights(os.path.join(work_dir, 'cat_dog.combined_model_weights.best.hdf5'))
#ret = new_model.predict(os.path.join(test_data_dir, "cat", "cat.983.jpg"))
#print ret
