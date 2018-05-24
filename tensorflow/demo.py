import tensorflow as tf 
import os
import numpy as np
from skimage import data, transform
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

def load_data(data_directory):
	directories = [d for d in os.listdir(data_directory)
				   if os.path.isdir(os.path.join(data_directory	, d))]

	labels = []
	images = []

	for d in directories:
		label_directory = os.path.join(data_directory, d)
		file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
		for f in file_names:
			images.append(data.imread(f))
			labels.append(int(d))
	return images, labels

ROOT_PATH = "F:/coding/python/Gaze-tracking"
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

images, labels = load_data(train_data_directory)
images28 = [transform.resize(image, (28, 28)) for image in images]
unique_labels = set(labels)


x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])


images_flat = tf.contrib.layers.flatten(x)

logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                    logits = logits))

train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


correct_pred = tf.argmax(logits, 1)


accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.set_random_seed(1234)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(201):
		_, loss_value = sess.run([train_op, loss], feed_dict={x: images28, y: labels})
		if (i%10 == 0):
			print("Loss: ", loss_value)



# print("images_flat: ", images_flat)
# print("logits: ", logits)
# print("loss: ", loss)
# print("predicted_labels: ", correct_pred)
# print (images, labels)

# images = np.array(images)
# labels = np.array(labels)
# print (images.ndim)
# print (images.size)
# print(len(set(labels)))

# traffic_signs = [300, 2250, 3650, 4000]



# for i in range(len(traffic_signs)):
# 	plt.subplot(1, 4, i+1)
# 	plt.axis('off')
# 	plt.imshow(images28[traffic_signs[i]], cmap="gray")
# 	plt.subplots_adjust(wspace = 0.5)
	
# 	print("shape: {0}, min: {1}, max: {2}".format(images28[traffic_signs[i]].shape, 
#                                                   images28[traffic_signs[i]].min(), 
#                                                   images28[traffic_signs[i]].max()))
# plt.show()
# plt.figure(figsize = (15, 15))
# i = 1

# for label in unique_labels:
# 	image = images28[labels.index(label)]
# 	plt.subplot(8,8,i)
# 	plt.axis('off')
# 	plt.title("Label {0} ({1})".format(label, labels.count(label)))
# 	i+=1
# 	plt.imshow(image)
