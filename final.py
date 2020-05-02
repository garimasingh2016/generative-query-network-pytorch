# code adapted from generative-query-network-pytorch Github repository created by Jesper Wohlert

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import random
    
# Load dataset
from shepardmetzler import ShepardMetzler
from torch.utils.data import DataLoader
from gqn import GenerativeQueryNetwork, partition
from PIL import Image

def deterministic_partition(images, viewpoints, indices):
    """
    Partition batch into context and query sets.
    :param images
    :param viewpoints
    :return: context images, context viewpoint, query image, query viewpoint
    """
    # Maximum number of context points to use
    _, b, m, *x_dims = images.shape
    _, b, m, *v_dims = viewpoints.shape

    # "Squeeze" the batch dimension
    images = images.view((-1, m, *x_dims))
    viewpoints = viewpoints.view((-1, m, *v_dims))

    # Partition into context and query sets
    context_idx, query_idx = indices[:-1], indices[-1]

    x, v = images[:, context_idx], viewpoints[:, context_idx]
    x_q, v_q = images[:, query_idx], viewpoints[:, query_idx]

    return x, v, x_q, v_q

dataset = ShepardMetzler("data/shepard_metzler_5_parts/") ## <= Choose your data location
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load model parameters onto CPU
state_dict = torch.load("./model-checkpoint.pth", map_location="cpu") ## <= Choose your model location

# Initialise new model with the settings of the trained one
model_settings = dict(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=8)
model = GenerativeQueryNetwork(**model_settings)

# Load trained parameters, un-dataparallel if needed
if True in ["module" in m for m in list(state_dict.keys())]:
    model = nn.DataParallel(model)
    model.load_state_dict(state_dict)
    model = model.module
else:
    model.load_state_dict(state_dict)

# Load data
train_images = []
train_labels = []
for scene_id in range(30):
	print("Loading scene " + str(scene_id))
	x, v = next(iter(loader))
	x_, v_ = x.squeeze(0), v.squeeze(0)

	for n in range(1, 15):
		# Sample a set of views
		n_context = n + 1
		print("\tGenerating reconstructions with context " + str(n_context))
		indices = random.sample([i for i in range(v_.size(1))], n_context)

		# Seperate into context and query sets
		x_c, v_c, x_q, v_q = deterministic_partition(x, v, indices)
		# get model reconstructions
		x_mu, r, kl = model(x_c[scene_id].unsqueeze(0), 
		                    v_c[scene_id].unsqueeze(0), 
		                    x_q[scene_id].unsqueeze(0),
		                    v_q[scene_id].unsqueeze(0))

		x_mu = x_mu.squeeze(0)
		r = r.squeeze(0)

		img = r.data.view(16, 16).numpy()
		train_images.append(img)
		train_labels.append(n_context)
train_images = np.array(train_images)
train_labels = np.array(train_labels)


# classification
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(16, 16)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(16, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("training and validation image classification")
train_history = model.fit(train_images, train_labels, validation_split=0.2, epochs=100)
print(train_history.history.keys())
# summarize history for accuracy
plt.figure()
plt.plot(train_history.history['accuracy'])
plt.plot(train_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.ylim(0, 1)
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("model-accuracy.png")
# summarize history for loss
plt.figure()
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.ylim(0, 4)
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("model-loss.png")


