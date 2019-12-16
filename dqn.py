import tensorflow as tf
import numpy as np
import time
from my_utils import stylePrint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
import os
import random

batch_size = 32
gamma = 0.99
initial_epsilon = 0.1
final_epsilon = 0.0001
explore = 1000000

# class Qnet(tf.keras.Model):
#     """A simple linear model."""

#     def __init__(self):
#         super(Qnet, self).__init__()
#         self.conv1 = Conv2D(32, 8, strides=4, padding='same', activation='relu')
#         self.pool1 = MaxPool2D(2, 2, padding='same')
#         self.conv2 = Conv2D(64, 4, strides=2, padding='same', activation='relu')
#         self.pool2 = MaxPool2D(2, 2, padding='same')
#         self.conv3 = Conv2D(64, 3, padding='same', activation='relu')
#         self.pool3 = MaxPool2D(2, 2, padding='same')
#         self.flatten = Flatten()
#         self.dense1 = Dense(256, activation='relu')
#         self.dense2 = Dense(2)

#     def call(self, x):
#         x = self.conv1(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.pool2(x)
#         x = self.conv3(x)
#         x = self.pool3(x)
#         x = self.flatten(x)
#         x = self.dense1(x)
#         x = self.dense2(x)
#         return self.l1(x)


def create_model():
    model = Sequential([
        Conv2D(32, 8, strides=4, padding='same', activation='relu', input_shape=(80, 80, 4)),
        MaxPool2D(2, 2, padding='same'),
        Conv2D(64, 4, strides=2, padding='same', activation='relu'),
        MaxPool2D(2, 2, padding='same'),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPool2D(2, 2, padding='same'),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(2)
    ])
    return model


model = create_model()
model_fix = create_model()
model_fix.trainable=False
model.summary()
# model.load_weights('./model/training_checkpoints')
# model_fix.load_weights('./model/training_checkpoints')

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.MeanSquaredError()

# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
# manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)


@tf.function
def train_step(q_target, s_t, a_t):
    a_t = tf.cast(a_t, tf.float32)
    with tf.GradientTape() as tape:
        q_network = tf.reduce_sum(model(s_t) * a_t, -1)
        loss = loss_object(q_target, q_network)
        # stylePrint(loss, fore='red')

    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss


def train(controller):
    minibatch = random.sample(controller['replayMemory'], batch_size)
    st1_batch = np.array([i[3] for i in minibatch])
    st_batch = np.zeros([batch_size, 80, 80, 4])
    at_batch = np.zeros([batch_size, 2])
    q_target_batch = np.zeros(batch_size)
    act1_batch = model_fix(st1_batch)

    for batch in range(0, len(minibatch)):
        st_batch[batch] = minibatch[batch][0]
        at_batch[batch] = minibatch[batch][1]
        terminal = minibatch[batch][4]
        if terminal:
            q_target_batch[batch] = minibatch[batch][2]
        else:
            q_target_batch[batch] = minibatch[batch][2] + gamma * np.max(act1_batch[batch])
    loss = train_step(q_target_batch, st_batch, at_batch)
    controller['stepLoss'].append(loss)
    if (controller['steps'] + 1) % 1000 == 0:
        model.save_weights('./model/training_checkpoints')
        model_fix.load_weights('./model/training_checkpoints')
        stylePrint('fix model result: ', model_fix(np.ones([1, 80, 80, 4])), fore='red', back='yellow')


def evaluate(controller, s_t):
    s_t = np.expand_dims(s_t, 0)
    action_ = model_fix(s_t)[0]

    if controller['istraining']:
        if controller['steps'] < explore:
            epsilon = initial_epsilon - controller['steps'] * (initial_epsilon - final_epsilon) / explore
        else:
            epsilon = 0

        if random.random() <= epsilon:
            action = np.random.choice([0, 1])
            stylePrint('random action: ', action, 'epsilon is :', epsilon, fore='red')
        else:
            action = np.argmax(action_)
            if action:
                stylePrint('action_:', action_, fore='red', back='yellow')
    return action, action_


def randomAction():
    return np.random.choice([0, 1], p=[1 - initial_epsilon, initial_epsilon])


# s_t = np.array([np.random.random([80, 80, 4]), np.random.random([80, 80, 4])])
# a_t = np.array([[0.0, 1.0], [0, 1]])
# r_t1 = 0.2
# s_t1 = np.random.random([80, 80, 4])
# terminate = 0
# # q_target = [0.4, 0.8]
# stylePrint(model(s_t), fore='red')
# stylePrint(model_fix(s_t), fore='yellow')
# print(train_step(q_target, s_t, a_t))