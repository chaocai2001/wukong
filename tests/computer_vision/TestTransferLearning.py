from wukong.computer_vision.TransferLearning import *

train_data_dir = r'../../samples/cat_dog/train/'
test_data_dir = r'../../samples/cat_dog/test'
work_dir = r'../../tmp'
model_name = "cat_dog"

create_default_deep_vision_model(work_dir, model_name, train_data_dir, test_data_dir)
