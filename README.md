Machine learning and deep learning are not the weapons just for the scientists. 

Wukong is the library to make you leverage machine learning/deep learning techiques easily and quickly.

By just one line code as the following, you can leverage the cutting edage deep learning technique--transfering learning to build a good image classification model, which can get amazing performance even training with a very little of samples.
```Python
from wukong.computer_vision.TransferLearning import *

create_default_deep_vision_model(work_dir=r'../../tmp', model_name="cat_dog", 
                                 train_data_dir=r'../../samples/cat_dog/train/', 
                                 test_data_dir=r'../../samples/cat_dog/test')

[Natural Language Understanding is coming soon]
