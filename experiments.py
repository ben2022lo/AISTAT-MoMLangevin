import gc
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import json

from CNN import create_model
from data import data_ge, results
from MoM import MoM


#data set
x_train, y_train_adv50, x_val, y_val, x_test, y_test = data_ge()
ar = [[1 if i == v else 0 for i in range(10)] for v in y_val]

#hyperparameters
k = 10
kmax = 40
nepc=50
bachsize = 1000
batchnb = len(x_train)/bachsize
nepcmom = nepc
mom_batch_size_trainning = bachsize
freq = nepcmom//nepc
b = 2
stability_for_bandit = 1
mediantype = 2

#SGD
model50,opti,l = create_model(optimizer="SGD", lr = 0.01)
model50.compile(loss=l, optimizer=opti, metrics=["accuracy"])
history50 = model50.fit(x_train, y_train_adv50, batch_size=bachsize, epochs=nepc, shuffle=True, validation_data=(x_val, y_val))
pred = model50.predict(x_test)
ypred = tf.math.argmax(pred, axis=1)
results(ypred, y_test)
model50.save('model50sgd.h5')
json_data = json.dumps(history50.history)
with open('history50sgd.history.json', 'w') as f:
    f.write(json_data)

#SGLD    
model50sgld,opti,l = create_model(optimizer="SGLD", lr = 0.01)
model50sgld.compile(loss=l, optimizer=opti, metrics=["accuracy"])
history50sgld = model50sgld.fit(x_train, y_train_adv50, batch_size=bachsize, epochs=nepc, shuffle=True, validation_data=(x_val, y_val))
pred = model50sgld.predict(x_test)
ypred = tf.math.argmax(pred, axis=1)
results(ypred, y_test)
model50sgld.save('model50sgld.h5')
json_data = json.dumps(history50sgld.history)
with open('history50sgld.history.json', 'w') as f:
    f.write(json_data)
    
#MoM SGD loss
gc.collect()
keras.backend.clear_session()
modelsgdloss,opti,l = create_model(optimizer="SGD", lr = 0.01)
momsgdloss = MoM(modelsgdloss,l,opti,x_train, y_train_adv50, x_val, y_val, ar, len(x_train)//k, mom_batch_size_trainning = mom_batch_size_trainning, epochs=nepcmom, freq = 1, dynamic = False,gradient_type = False, mediantype = mediantype)
pred = modelsgdloss.predict(x_test)
ypred = tf.math.argmax(pred, axis=1)
results(ypred, y_test)
modelsgdloss.save('modelsgdloss.h5')
momsgdloss["usage"] = momsgdloss["usage"].tolist()
json_data = json.dumps(momsgdloss)
with open('momsgdloss.json', 'w') as f:
    f.write(json_data)

#MoM SGD grad
gc.collect()
keras.backend.clear_session()
modelsgdgrad,opti,l = create_model(optimizer="SGD", lr = 0.01)
momsgdgrad = MoM(modelsgdgrad,l,opti,x_train, y_train_adv50, x_val, y_val, ar, len(x_train)//k, mom_batch_size_trainning = mom_batch_size_trainning, epochs=nepcmom, freq = 1, dynamic = False,gradient_type = True, mediantype = mediantype)
pred = modelsgdgrad.predict(x_test)
ypred = tf.math.argmax(pred, axis=1)
results(ypred, y_test)

modelsgdgrad.save('modelsgdgrad.h5')
momsgdgrad["usage"] = momsgdgrad["usage"].tolist()
json_data = json.dumps(momsgdgrad)
with open('momsgdgrad.json', 'w') as f:
    f.write(json_data)

# MoM SGLD loss
gc.collect()
keras.backend.clear_session()
modelsgldloss,opti,l = create_model(optimizer="SGLD", lr = 0.01)
momsgldloss = MoM(modelsgldloss,l,opti,x_train, y_train_adv50, x_val, y_val, ar, len(x_train)//k, mom_batch_size_trainning = mom_batch_size_trainning, epochs=nepcmom, freq = 1, dynamic = False,gradient_type = False, mediantype = mediantype)
pred = modelsgldloss.predict(x_test)
ypred = tf.math.argmax(pred, axis=1)
results(ypred, y_test) 

modelsgldloss.save('modelsgldloss.h5')
momsgldloss["usage"] = momsgldloss["usage"].tolist()
json_data = json.dumps(momsgldloss)
with open('momsgldloss.json', 'w') as f:
    f.write(json_data)

# MoM SGLD grad
gc.collect()
keras.backend.clear_session()
modelsgldgrad,opti,l = create_model(optimizer="SGLD", lr = 0.01)
momsgldgrad = MoM(modelsgldgrad,l,opti,x_train, y_train_adv50, x_val, y_val, ar, len(x_train)//k, mom_batch_size_trainning = mom_batch_size_trainning, epochs=nepcmom, freq = 1, dynamic = False,gradient_type = True, mediantype = mediantype)
modelsgldgrad.save('modelsgldgrad.h5')
momsgldgrad["usage"] = momsgldgrad["usage"].tolist()
json_data = json.dumps(momsgldgrad)
with open('momsgldgrad.json', 'w') as f:
    f.write(json_data)
pred = modelsgldgrad.predict(x_test)
ypred = tf.math.argmax(pred, axis=1)
results(ypred, y_test)


# results display
fig, ax = plt.subplots(figsize = (12,6))
try:
  ax.plot(history50["val_accuracy"],label="SGD")
except:
  pass
try:
  ax.plot(history50sgld["val_accuracy"],label="SGLD")
except:
  pass
try:
  ax.plot(momsgdloss["val_accuracy"],label="MoM SGD loss")
except:
  pass
try:
  ax.plot(momsgdgrad["val_accuracy"],label="MoM SGD grad")
except:
  pass
try:
  ax.plot(momsgldloss["val_accuracy"],label="MoM SGLD loss")
except:
  pass

try:
  ax.plot(momsgldgrad["val_accuracy"],label="MoM SGLD grad")
except:
  pass
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy on validation data")
plt.legend()


# usage frequency for detection
try:
  plt.figure(figsize = (14,4))
  plt.bar(list(range(len(x_train)))[:], momsgldgrad["usage"][:])
  plt.xlabel("id")
  plt.ylabel("usage")
  plt.title("MOM SGLD grad usage")
except:
  pass
try:
  print(sum(momsgldgrad["usage"][:len(momsgldgrad["usage"])//2])/(len(momsgldgrad["usage"])//2),sum(momsgldgrad["usage"][len(momsgldgrad["usage"])//2:])/(len(momsgldgrad["usage"])//2))
except:
  pass