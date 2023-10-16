import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.initializers import RandomUniform
from SGLD import SGLD

def create_model(optimizer="SGD", lr=0.01):
  '''
  optimizer (str) : SGD or SGLD
  '''

  input = Input(shape=(28,28,1))
  norm = BatchNormalization()(input)
  Conv0 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05))(norm)
  Max0 = MaxPooling2D((2, 2))(Conv0)
  Conv1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05))(Max0)
  Max1 = MaxPooling2D((2, 2))(Conv1)
  Conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05))(Max1)
  Max2 = MaxPooling2D((2, 2))(Conv2)
  flat = Flatten()(Max2)
  dense = Dense(64, activation='relu')(flat)
  output = Dense(10, activation='softmax')(dense)

  model = Model(inputs=input, outputs=output)

  l = tf.keras.losses.SparseCategoricalCrossentropy(
      reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
      name='sparse_categorical_crossentropy'
  )

  if optimizer == "SGD":
    opti = tf.keras.optimizers.SGD(learning_rate=lr)
    print("net with Stochastic Gradient Descent")
  elif optimizer == "SGLD":
    opti = SGLD(learning_rate=lr)
    print("net with Stochastic Langevin Dynamics")
  else:
    print("Chose an appropriate optimizer (SGD or SGLD)")
    return None

  return model,opti,l