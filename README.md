Machine learning and deep learning are not the weapons just for the scientists. 

Wukong is the library to make you leverage machine learning/deep learning techiques easily and quickly.

By just several line codes as the following, you can leverage the cutting edage deep learning technique--transfering learning to build a good image classification model, which can get amazing performance even training with a very little of samples.
```Python
from wukong.computer_vision.TransferLearning import WuKongVisionModel

train_data_dir = r'../../samples/cat_dog/train/'
test_data_dir = r'../../samples/cat_dog/test'
work_dir = r'../../tmp'
task_name = "cat_dog"

# train a model with the default configuration
model = WuKongVisionModel()
model.train_for_new_task(work_dir, task_name, train_data_dir, test_data_dir)

# predict by the trained model
ret = model.predict(os.path.join(test_data_dir, "cat", "cat.983.jpg"))
print ret
```
[Natural Language Understanding is coming soon]


# Transfer Learning 

DNN has shown the very significant performance in different areas. But, it is not easy to leverage it in your cases. When leveraging DNN to solve your real problems, you are always challenged by the following hard problems.

**1 Lack of labeled data**

As you known, training a DNN is hard. So many weights need to be adjusted in a DNN, for training a DNN normally needs huge amount of training samples, such as, millions of training samples.
In real world, you might not have enough labeled data to train a DNN, and labeling data is always expensive.

**2 No Bible for constructing DNN**

Deep learning is still in its early stage. Although, some typical DNN structures have been proved being successful in some areas, it is still no mature theory or methodology to decide the detail structure of DNN. For example, we know CNN (Conventional Neural Network) is a good choice for computer vision problems, but there is not mature theory or methodology on deciding the detail structure of the CNN, Such as, the number of layers. It is a time consuming task to experiment the different DNN structures.

Transfer learning is to address the above problem. Transfer learning is about reusing a successful DNN in similar cases. With transfer learning, the original DNN would work for new cases well after training with small amount of new training samples.

## Why transfer learning works?
Like CNN, the lower layers of convolutional neural networks have long been known to resemble conventional computer vision features like edge detectors and Gabor filters. And the top hidden layers are specialized to the task it was trained for.
Transfer learning is to leverage the DNN’s feature extracting capability to handle new problems. By replacing the original top layers with the new top layers, the reconstructed DNN would be used to handle the new task. And by leveraging the feature extracting capability of the trained layers, only very small amount of labeled data is needed to train the new top layers and tune the trained DNN finely for the new task

## How to use transfer learning?
* Remove the top layer of the original DNN
* Build the new top layers on the original DNN
* Train the new top layers
* Tune the new DNN finely 

Wukong is a library, which encapsulate the transfer learning process. With the libraray, it is quite easy to practice transfer learning, you just need to pass your pretrained DNN model and top model to Wukong, and then Wukong would take over all of the left jobs. 
Even, you can leverage the default pretrained model and top model in Wukong, the default model works very well for many image classification problems. 








