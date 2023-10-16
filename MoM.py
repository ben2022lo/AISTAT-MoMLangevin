import numpy as np
import tensorflow as tf
from tqdm import tqdm
from geom_median.numpy import compute_geometric_median
from math import log,exp
import tensorflow_probability as tfp



def Tukey(gradients):
  '''
  approximate tukey median for a list of gradients
  '''
  l = len(gradients[0])
  mus = gradients
  depths = []
  for mu in mus:
    Xs = [[tf.subtract(tensor1, tensor2) for tensor1, tensor2 in zip(gradient, mu)] for gradient in gradients]
    vs = [[tf.subtract(tensor1, tensor2) for tensor1, tensor2 in zip(gradient, mu)] for gradient in gradients]
    ps = []
    for v in vs:
      p = 0
      for X in Xs:
        if tf.reduce_sum([tf.reduce_sum(tf.multiply(X[i], v[i])) for i in range(l)]) >= 0:
          p += 1
      ps.append(p)
    depths.append(min(ps))
  ind = tf.argmax(depths)
  return gradients[ind], ind

def geometric(gradients):
  '''
    geometric median for a list of gradients
  '''
  for i in range(len(gradients)):
    for j in range(len(gradients[0])):
      gradients[i][j] = np.array(gradients[i][j])
  out = compute_geometric_median(gradients, weights=None)

  geo_med_grad = []
  for g in out.median:
    ten = tf.convert_to_tensor(g, dtype=np.float32)
    geo_med_grad.append(ten)
  return geo_med_grad

def median_grad(grad_list, mediantype):
  '''
      calculate the median of gradients
  '''
  if mediantype ==1:
    return (geometric(grad_list), -1)
  elif mediantype == 2:
    grad, ind = Tukey(grad_list)
    return (grad, ind)

def med(model,l,opti,data,labels,fulldata,fulllabels,usage,i,batch_size,mom_batch_size_trainning,initial_indexes,loss_epoch,accuracy_epoch,val_loss_epoch,val_accuracy_epoch,percent,gradient_type = False, mediantype = 0):
    '''
        calculate the median of gradients or the median of losses and return the indexes of images used for update
    '''
    loss = None
    def f(i):
      return(tf.reduce_mean(loss[i*batch_size: tf.math.minimum(len(data),(i+1)*batch_size)]))
    if gradient_type == False:
      predi = model(data)
      loss = tf.keras.losses.sparse_categorical_crossentropy(labels ,predi, from_logits=True)
      batch_mean_loss=tf.vectorized_map(f,tf.range(0, len(data)//batch_size))
      median = tfp.stats.percentile(batch_mean_loss, percent)
      closest_batch_index = tf.argmin(tf.abs(batch_mean_loss - median))
      closest_batch_data = data[closest_batch_index*batch_size:min(len(data),(closest_batch_index+1)*batch_size)]
      closest_batch_labels = labels[closest_batch_index*batch_size:min(len(data),(closest_batch_index+1)*batch_size)]
      closest_batch_indexes = initial_indexes[closest_batch_index*batch_size:min(len(data),(closest_batch_index+1)*batch_size)]
      closest_batch_indexes = tf.reshape(closest_batch_indexes, (batch_size, 1))
      if mom_batch_size_trainning ==0 or mom_batch_size_trainning >= batch_size:
        with tf.GradientTape() as tape:
          predictions = model(closest_batch_data)
          loss_value = l(closest_batch_labels, predictions)
        gradients = tape.gradient(loss_value, model.trainable_weights)
        opti.apply_gradients(zip(gradients, model.trainable_weights))
      else:
        for j in range(batch_size//mom_batch_size_trainning):
          with tf.GradientTape() as tape:
            predictions = model(closest_batch_data[j*mom_batch_size_trainning:(j+1)*mom_batch_size_trainning])
            loss_value = l(closest_batch_labels[j*mom_batch_size_trainning:(j+1)*mom_batch_size_trainning], predictions)
          gradients = tape.gradient(loss_value, model.trainable_weights)
          opti.apply_gradients(zip(gradients, model.trainable_weights))
      u = tf.tensor_scatter_nd_add(usage, closest_batch_indexes, tf.ones((batch_size,), dtype=tf.int32))
    else:
      lst = []
      #print(len(data),batch_size,len(data)//batch_size)
      for j in range(len(data)//batch_size):
          with tf.GradientTape() as tape:
            predictions = model(data[j*batch_size:(j+1)*batch_size])
            loss_value = l(labels[j*batch_size:(j+1)*batch_size], predictions)
          gradients = tape.gradient(loss_value, model.trainable_weights)
          lst.append(gradients)
      if len(lst) == 1:
        print("pas normal")
        gra = lst[0]
        u = tf.tensor_scatter_nd_add(usage, tf.reshape(initial_indexes, (tf.shape(initial_indexes)[0], 1)), tf.ones((batch_size,), dtype=tf.int32))
      else:
        gra,ind = median_grad(lst, mediantype)
        if ind !=-1:
          closest_batch_indexes = initial_indexes[ind*batch_size:min(len(data),(ind+1)*batch_size)]
          closest_batch_indexes = tf.reshape(closest_batch_indexes, (batch_size, 1))
          u = tf.tensor_scatter_nd_add(usage, closest_batch_indexes, tf.ones((batch_size,), dtype=tf.int32))
        else:
          u = None
      opti.apply_gradients(zip(gra, model.trainable_weights))
    return(u)
def MoM(model,l,opti,data, labels, x_val, y_val, ar, batch_size = 1000, mom_batch_size_trainning = 75, epochs = 10, freq = 1, dynamic = False, nu = None, gradient_type = False, mediantype = 0):
    '''
         MoM based gradient descent with bandit option to choose the parameter K
    '''
    if not(dynamic):
      initial_indexes = tf.range(0, len(data))#tf.convert_to_tensor(np.array(list(range(len(data)))))
      usage = tf.zeros(len(data), dtype=tf.int32)
      loss_epoch =  []
      accuracy_epoch =  []
      val_loss_epoch = []
      val_accuracy_epoch = []
      loss = None
      def f(i):
        return(tf.reduce_mean(loss[i*batch_size: tf.math.minimum(len(data),(i+1)*batch_size)]))
      for i in range(epochs):
        print("Epoch ", i+1)

        initial_indexes = tf.random.shuffle(initial_indexes)
        shuffled_data = tf.gather(data, initial_indexes)
        shuffled_labels = tf.gather(labels, initial_indexes)
        """plt.imshow(shuffled_data[list(initial_indexes).index(2)])
        plt.show()
        print(shuffled_labels[list(initial_indexes).index(2)])"""
        if mom_batch_size_trainning == 0 or mom_batch_size_trainning>=len(shuffled_data):
          #print(" mom batch to big")
          usage = med(
              model,
              l,
              opti,
              shuffled_data,
              shuffled_labels,
              data,
              labels,
              usage,
              i,
              batch_size,
              mom_batch_size_trainning,
              initial_indexes,
              loss_epoch,
              accuracy_epoch,
              val_loss_epoch,
              val_accuracy_epoch,
              50,
              gradient_type = gradient_type,
              mediantype = mediantype
              )
        else:
          for ibatch in tqdm(range(len(shuffled_data)//mom_batch_size_trainning)):
            #print("ibatch",ibatch)
            #print(len(initial_indexes[ibatch*mom_batch_size_trainning:(ibatch+1)*mom_batch_size_trainning]))
            #print(len(shuffled_data[ibatch*mom_batch_size_trainning:(ibatch+1)*mom_batch_size_trainning]))
            usage = med(
                model,
                l,
                opti,
                shuffled_data[ibatch*mom_batch_size_trainning:(ibatch+1)*mom_batch_size_trainning],
                shuffled_labels[ibatch*mom_batch_size_trainning:(ibatch+1)*mom_batch_size_trainning],
                data,
                labels,
                usage,
                i,
                batch_size//(len(shuffled_data)//mom_batch_size_trainning),
                mom_batch_size_trainning,
                initial_indexes[ibatch*mom_batch_size_trainning:(ibatch+1)*mom_batch_size_trainning],
                loss_epoch,
                accuracy_epoch,
                val_loss_epoch,
                val_accuracy_epoch,
                50,
                gradient_type = gradient_type,
                mediantype = mediantype
                )
        if i % freq == 0:
          pre=model(data)
          preval=model(x_val)
          loss_epoch.append(float(
              tf.reduce_mean(
                  l(
                      labels ,
                      pre
                      )
              )

          ))
          fular = [[1 if i == v else 0 for i in range(10)] for v in labels]
          accuracy_epoch.append(float(
              tf.keras.metrics.CategoricalAccuracy()(
                  fular,
                  pre
              )
          ))
          val_loss_epoch.append(float(
              tf.reduce_mean(
                  l(
                      y_val,
                      preval
                      )
              )
          ))
          val_accuracy_epoch.append(float(
              tf.keras.metrics.CategoricalAccuracy()(
                  ar,
                  preval
              )
          ))
          print("tac:",accuracy_epoch[len(accuracy_epoch)-1], loss_epoch[len(accuracy_epoch)-1])
          print("vac:",val_accuracy_epoch[len(val_accuracy_epoch)-1], val_loss_epoch[len(val_accuracy_epoch)-1])
      if usage != None:
        return {"loss" : loss_epoch, "accuracy" : accuracy_epoch, "val_loss" : val_loss_epoch, "val_accuracy" : val_accuracy_epoch,"usage":usage.numpy()}
      else:
        return {"loss" : loss_epoch, "accuracy" : accuracy_epoch, "val_loss" : val_loss_epoch, "val_accuracy" : val_accuracy_epoch}
    '''
    else:
      newk = k
      newpercent = 50
      newsize = 9*len(data)//10
      newbatch_size = newsize//newk
      initial_indexes = tf.range(0, len(data))#tf.convert_to_tensor(np.array(list(range(len(data)))))
      usage = tf.zeros(len(data), dtype=tf.int32)
      nk = int(log(min(newsize,kmax),b)+1)
      npercent = 1
      p = tf.ones((nk,npercent), dtype=tf.float64)/(nk*npercent)
      ps = tf.ones((nk,npercent), dtype=tf.float64)
      lhat = tf.zeros((nk,npercent), dtype=tf.float64)

      loss_epoch =  []
      accuracy_epoch =  []
      val_loss_epoch = []
      val_accuracy_epoch = []
      loss = None
      khisto = []
      perchisto = []
      prefperhisto = [[] for i in range(npercent)]
      prefkhisto = [[] for i in range(nk)]
      if nu == None:
        nu = sqrt((log(nk*npercent))/((nk*npercent)*(epochs//stability_for_bandit)))
      def f(i):
        return(tf.reduce_mean(loss[i*newbatch_size: tf.math.minimum(newsize,(i+1)*newbatch_size)]))
      for i in range(epochs//stability_for_bandit):
        gc.collect()
        keras.backend.clear_session()
        print("Epoch ", i*stability_for_bandit+1)
        tbegin = time()
        initial_indexes = tf.random.shuffle(initial_indexes)
        shuffled_data = tf.gather(data, initial_indexes)
        shuffled_labels = tf.gather(labels, initial_indexes)
        #print(p)
        s0 =tf.math.reduce_sum(p,axis = 0)
        s1 = tf.math.reduce_sum(p,axis = 1)
        print(s0)
        print(s1)
        samples = tf.random.categorical(tf.math.log([tf.reshape(p, (nk*npercent,))]), 1)
        #print(samples)
        ki = int(samples)//npercent
        percenti = int(samples)%npercent
        #print(ki,percenti)
        newk=int(b**ki)
        newpercent=(100/(npercent+1))*(1+percenti)
        print(newk,newpercent)
        khisto.append(ki)
        perchisto.append(newpercent)
        for loop in range(npercent):
          prefperhisto[loop].append(float(s0[loop]))
        for loop in range(nk):
          prefkhisto[loop].append(float(s1[loop]))

        newbatch_size = newsize//newk

        testpredi =  model(shuffled_data[newsize:])
        inittestlosstesor = tf.keras.losses.sparse_categorical_crossentropy(shuffled_labels[newsize:] ,testpredi, from_logits=True)
        intittestloss = tf.reduce_mean(inittestlosstesor)
        if mom_batch_size_trainning == 0 or mom_batch_size_trainning>=newsize:
          usage = med(model,
                      l,
                      opti,
                      shuffled_data[:newsize],
                      shuffled_labels[:newsize],
                      data,
                      labels,
                      usage,
                      i*stability_for_bandit,
                      newbatch_size,
                      mom_batch_size_trainning,
                      initial_indexes[:newsize],
                      loss_epoch,
                      accuracy_epoch,
                      val_loss_epoch,
                      val_accuracy_epoch,
                      newpercent,
                      gradient_type = gradient_type,
                      mediantype = mediantype
                      )
        else:
          for ibatch in range(newsize//mom_batch_size_trainning):
            #print("ibatch",ibatch)
            #print(len(shuffled_data[ibatch*mom_batch_size_trainning:(ibatch+1)*mom_batch_size_trainning]))
            usage = med(model,
                        l,
                        opti,
                        shuffled_data[ibatch*mom_batch_size_trainning:(ibatch+1)*mom_batch_size_trainning],
                        shuffled_labels[ibatch*mom_batch_size_trainning:(ibatch+1)*mom_batch_size_trainning],
                        data,
                        labels,
                        usage,
                        i*stability_for_bandit,
                        mom_batch_size_trainning//newk,
                        mom_batch_size_trainning,
                        initial_indexes[ibatch*mom_batch_size_trainning:(ibatch+1)*mom_batch_size_trainning],
                        loss_epoch,
                        accuracy_epoch,
                        val_loss_epoch,
                        val_accuracy_epoch,
                        newpercent,
                        gradient_type = gradient_type,
                        mediantype = mediantype
                        )
        if i*stability_for_bandit % freq == 0:
          pre=model(data)
          preval=model(x_val)
          loss_epoch.append(float(
              tf.reduce_mean(
                  l(
                      labels ,
                      pre
                      )
              )

          ))
          fular = [[1 if i == v else 0 for i in range(outputsize)] for v in labels]
          accuracy_epoch.append(float(
              tf.keras.metrics.CategoricalAccuracy()(
                  fular,
                  pre
              )
          ))
          val_loss_epoch.append(float(
              tf.reduce_mean(
                  l(
                      y_val,
                      preval
                      )
              )
          ))
          val_accuracy_epoch.append(float(
              tf.keras.metrics.CategoricalAccuracy()(
                  ar,
                  preval
              )
          ))
          print("tac:",accuracy_epoch[len(accuracy_epoch)-1], loss_epoch[len(accuracy_epoch)-1])
          print("vac:",val_accuracy_epoch[len(val_accuracy_epoch)-1], val_loss_epoch[len(val_accuracy_epoch)-1])


        for j in range(stability_for_bandit-1):
          print("Epoch ", i*stability_for_bandit+j+2)
          init_indexes = tf.random.shuffle(initial_indexes[:newsize])
          sh_data=tf.gather(data, init_indexes)
          sh_labels=tf.gather(labels, init_indexes)
          #print(len(init_indexes))
          if mom_batch_size_trainning == 0 or mom_batch_size_trainning>=newsize:
            usage = med(model,
                        l,
                        opti,
                        sh_data,
                        sh_labels,
                        data,
                        labels,
                        usage,
                        i*stability_for_bandit+j+1,
                        newbatch_size,
                        mom_batch_size_trainning,
                        init_indexes,
                        loss_epoch,
                        accuracy_epoch,
                        val_loss_epoch,
                        val_accuracy_epoch,
                        newpercent,
                        gradient_type = gradient_type,
                        mediantype = mediantype
                        )
          else:
            for ibatch in range(newsize//mom_batch_size_trainning):
              #print(len(sh_data[ibatch*mom_batch_size_trainning:(ibatch+1)*mom_batch_size_trainning]))
              usage = med(model,
                        l,
                        opti,
                        sh_data[ibatch*mom_batch_size_trainning:(ibatch+1)*mom_batch_size_trainning],
                        sh_labels[ibatch*mom_batch_size_trainning:(ibatch+1)*mom_batch_size_trainning],
                        data,
                        labels,
                        usage,
                        i*stability_for_bandit+j+1,
                        mom_batch_size_trainning//newk,
                        mom_batch_size_trainning,
                        init_indexes[ibatch*mom_batch_size_trainning:(ibatch+1)*mom_batch_size_trainning],
                        loss_epoch,
                        accuracy_epoch,
                        val_loss_epoch,
                        val_accuracy_epoch,
                        newpercent,
                        gradient_type = gradient_type,
                        mediantype = mediantype
                        )
          if i*stability_for_bandit+j+1 % freq == 0:
            pre=model(data)
            preval=model(x_val)
            loss_epoch.append(float(
                tf.reduce_mean(
                    l(
                        labels ,
                        pre
                        )
                )

            ))
            fular = [[1 if i == v else 0 for i in range(outputsize)] for v in labels]
            accuracy_epoch.append(float(
                tf.keras.metrics.CategoricalAccuracy()(
                    fular,
                    pre
                )
            ))
            val_loss_epoch.append(float(
                tf.reduce_mean(
                    l(
                        y_val,
                        preval
                        )
                )
            ))
            val_accuracy_epoch.append(float(
                tf.keras.metrics.CategoricalAccuracy()(
                    ar,
                    preval
                )
            ))
            print("tac:",accuracy_epoch[len(accuracy_epoch)-1], loss_epoch[len(accuracy_epoch)-1])
            print("vac:",val_accuracy_epoch[len(val_accuracy_epoch)-1], val_loss_epoch[len(val_accuracy_epoch)-1])

        testpredi = model(shuffled_data[newsize:])
        finaltestlosstesor = tf.keras.losses.sparse_categorical_crossentropy(shuffled_labels[newsize:] ,testpredi, from_logits=True)
        finaltestloss = tf.reduce_mean(finaltestlosstesor)


        tcost = time()
        duration =((tcost-tbegin)/stability_for_bandit)
        winrate = (finaltestloss/intittestloss)**(1/stability_for_bandit)
        print(duration,winrate,winrate**4)
        cost = 1-exp(-duration/winrate**4)
        print(cost)

        """c = log(1.005)
        r = 1
        truecost = -log(r*winrate)#-log(winrate/duration)
        truecostnormalizedandofseted =(min(c,max(-c,truecost))+c)/(2*c)
        cost =truecostnormalizedandofseted#100*truecost#truecostnormalizedandofseted#100*truecost#ttruecostnormalizedandofseted# truecostofseted
        print(duration,winrate,truecost,cost)"""
        lhat = tf.tensor_scatter_nd_add(lhat,tf.constant([[ki,percenti]]),tf.reshape(cost/p[ki,percenti],[1]))
        ps =tf.tensor_scatter_nd_update(ps,tf.constant([[ki,percenti]]),tf.reshape(tf.math.exp(-nu*lhat[ki,percenti]),[1]))
        p = ps/tf.reduce_sum(ps)
      print(tf.math.reduce_sum(p,axis = 0))
      print(tf.math.reduce_sum(p,axis = 1))
      print(p)
      if usage != None:
        return {"loss" : loss_epoch,
                "accuracy" : accuracy_epoch,
                "val_loss" : val_loss_epoch,
                "val_accuracy" : val_accuracy_epoch,
                "usage":usage.numpy(),
                "k":khisto,
                "per":perchisto,
                "prefk":prefkhisto,
                "prefper": prefperhisto
                }
      else:
        return {"loss" : loss_epoch,
                "accuracy" : accuracy_epoch,
                "val_loss" : val_loss_epoch,
                "val_accuracy" : val_accuracy_epoch,
                "k":khisto,
                "per":perchisto,
                "prefk":prefkhisto,
                "prefper": prefperhisto
                }
    '''