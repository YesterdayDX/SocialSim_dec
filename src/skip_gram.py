import collections
import math
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from training_index import training_index

# PLATFORM = 'Twitter'
# TOPIC = 'Cyber'



# if TOPIC == 'Cyber':
#   N_TOP_ROOT = 20000
#   N_TOP_COMM = 30000
#   batch_size = 128
#   embed_size = 128
#   num_sample = 64
# elif TOPIC == 'Cve':
#   N_TOP_ROOT = 100
#   N_TOP_COMM = 2000
#   batch_size = 32
#   embed_size = 128
#   num_sample = 64
# elif TOPIC == 'Crypto':
#   N_TOP_ROOT = 4000
#   N_TOP_COMM = 6000
#   batch_size = 64
#   embed_size = 128
#   num_sample = 64
# elif TOPIC == 'CveS2':
#   N_TOP_ROOT = 93
#   N_TOP_COMM = 375
#   batch_size = 32
#   embed_size = 128
#   num_sample = 64


class skip_gram:
  def __init__(self, root_size, comm_size, batch_size, embedding_size, pre_window, num_sample, log_file):
    '''
    target_user --> embedding_in --> embeded_vec --> embedding_out --> output
    '''
    self.root_size = root_size
    self.comm_size = comm_size
    self.batch_size = batch_size
    self.embedding_size = embedding_size
    self.pre_window = pre_window
    self.num_sample = num_sample
    self.log_file = log_file
    self.weights = [0.25, 0.25, 0.25, 0.25]
    self.valid_examples = np.random.choice(10, 10, replace=False) + self.root_size

    self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    self.train_labels = tf.placeholder(tf.int32, shape=[batch_size, pre_window+1])
    self.negative_samples = tf.placeholder(tf.int32, shape=[num_sample])
    self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

    self.embeddings_in = tf.Variable(
      tf.random_uniform([comm_size, embedding_size], -1.0, 1.0))
    self.embeddings_out = tf.Variable(
      tf.random_uniform([(root_size+comm_size), embedding_size], -1.0, 1.0))

    self.embed_in = tf.nn.embedding_lookup(self.embeddings_in, self.train_inputs)
    # self.embed_out = tf.nn.embedding_lookup(self.embeddings_out, tf.reshape(self.train_labels, [-1]))
    self.embed_out_smp = tf.nn.embedding_lookup(self.embeddings_out, self.negative_samples)

    # Compute the cosine simularity between minibatch adn all embeddings
    norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings_in), 1, keepdims=True))
    normalized_embeddings_in = self.embeddings_in / norm
    norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings_out), 1, keepdims=True))
    normalized_embeddings_out = self.embeddings_out / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings_out, self.valid_dataset)
    self.similarity = tf.matmul(valid_embeddings, normalized_embeddings_in, transpose_b=True)

    RLE1 = 0
    for i in range(self.pre_window+1):
      embed_out = tf.nn.embedding_lookup(self.embeddings_out, tf.reshape(self.train_labels[:,i], [-1]))
      E1 = tf.matmul(self.embed_in, embed_out, transpose_b=True)
      LE1 = tf.log(tf.sigmoid(tf.diag_part(E1)))
      RLE1 += self.weights[i]*tf.reduce_mean(LE1)

    E2 = tf.matmul(self.embed_in, self.embed_out_smp, transpose_b=True)
    LE2 = tf.log(tf.nn.sigmoid(-E2))
    RLE2 = tf.reduce_mean(LE2)

    self.loss = -(RLE1+RLE2)

  def train(self, X, Y, PLATFORM, TOPIC, learn_rate=1.0, Total_steps=500000):
    # optimizer = tf.train.AdamOptimizer(learn_rate).minimize(self.loss)
    optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.loss)
    train_index = training_index(X.shape[0])

    saver = tf.train.Saver()
    # config = tf.ConfigProto(device_count={"CPU": 40},
    #         inter_op_parallelism_threads = 40,
    #         intra_op_parallelism_threads = 40,
    #         log_device_placement=True)

    avg_loss = 0
    avg_loss_list = list()
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for step in range(Total_steps):
        index = train_index.next(self.batch_size)

        neg_samples = self.negative_sampling(self.root_size+self.comm_size, self.num_sample)
        feed_dict={self.train_inputs: X[index], self.train_labels: Y[index], self.negative_samples:  neg_samples}

        _, loss = sess.run([optimizer, self.loss], feed_dict=feed_dict)
        avg_loss += loss

        if step%2000 == 0:
          if step>0:
            avg_loss /= 2000
            avg_loss_list.append(avg_loss)
          print("Average loss at step ", step, ": ", avg_loss)
          avg_loss = 0

          if step % 10000 == 0:
            sim = self.similarity.eval()
            for i in xrange(len(self.valid_examples)):
              top_k = 8  # number of nearest neighbors
              nearest = (-sim[i, :]).argsort()[1:top_k + 1]
              log_str = 'Nearest to %d:' % self.valid_examples[i]
              for k in xrange(top_k):
              # print nearest[k]
              # close_word = reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, str(nearest[k]))
              # print(log_str)

      embeddings_in = self.embeddings_in.eval()
      embeddings_out = self.embeddings_out.eval()
      print embeddings_in.shape, embeddings_out.shape

      norm = np.sqrt(np.sum(np.square(embeddings_out), 1, keepdims=True))
      embeddings_out = embeddings_out / norm
      norm = np.sqrt(np.sum(np.square(embeddings_in), 1, keepdims=True))
      embeddings_in = embeddings_in / norm

      np.savetxt('./results/'+PLATFORM+'/'+TOPIC+'/embeddings_in.csv', embeddings_in)
      np.savetxt('./results/'+PLATFORM+'/'+TOPIC+'/embeddings_out.csv', embeddings_out)
      np.savetxt('./results/'+PLATFORM+'/'+TOPIC+'/avg_loss.csv', avg_loss_list)

      saver.save(sess, self.log_file)

  def retrain(self, X, Y, PLATFORM, TOPIC, learn_rate=1.0, Total_steps=500000):
    # optimizer = tf.train.AdamOptimizer(learn_rate).minimize(self.loss)
    optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.loss)
    train_index = training_index(X.shape[0])

    saver = tf.train.Saver()
    # config = tf.ConfigProto(device_count={"CPU": 40},
    #         inter_op_parallelism_threads = 40,
    #         intra_op_parallelism_threads = 40,
    #         log_device_placement=True)

    avg_loss = 0
    avg_loss_list =list(np.loadtxt('../results/'+PLATFORM+'/'+TOPIC+'/avg_loss.csv'))
    with tf.Session() as sess:
      saver.restore(sess, self.log_file)
      for step in range(Total_steps):
        index = train_index.next(self.batch_size)

        neg_samples = self.negative_sampling(self.root_size+self.comm_size, self.num_sample)
        feed_dict={self.train_inputs: X[index], self.train_labels: Y[index], self.negative_samples:  neg_samples}

        _, loss = sess.run([optimizer, self.loss], feed_dict=feed_dict)
        avg_loss += loss

        if step%2000 == 0:
          if step>0:
            avg_loss /= 2000
            avg_loss_list.append(avg_loss)
          print("Average loss at step ", step, ": ", avg_loss)
          avg_loss = 0

          if step % 10000 == 0:
            sim = self.similarity.eval()
            for i in xrange(len(self.valid_examples)):
              top_k = 8  # number of nearest neighbors
              nearest = (-sim[i, :]).argsort()[1:top_k + 1]
              log_str = 'Nearest to %d:' % self.valid_examples[i]
              for k in xrange(top_k):
              # print nearest[k]
              # close_word = reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, str(nearest[k]))
              print(log_str)

      embeddings_in = self.embeddings_in.eval()
      embeddings_out = self.embeddings_out.eval()
      print embeddings_in.shape, embeddings_out.shape

      norm = np.sqrt(np.sum(np.square(embeddings_out), 1, keepdims=True))
      embeddings_out = embeddings_out / norm
      norm = np.sqrt(np.sum(np.square(embeddings_in), 1, keepdims=True))
      embeddings_in = embeddings_in / norm

      np.savetxt('./results/'+PLATFORM+'/'+TOPIC+'/embeddings_in.csv', embeddings_in)
      np.savetxt('./results/'+PLATFORM+'/'+TOPIC+'/embeddings_out.csv', embeddings_out)
      np.savetxt('./results/'+PLATFORM+'/'+TOPIC+'/avg_loss.csv', avg_loss_list)

      saver.save(sess, self.log_file)

  def load_data(self, platform, topic):
    print 'Loading data: '+platform+'_'+topic + '  ROOR_TOP=%d  COMM_TOP=%d' %(self.root_size, self.comm_size)
    FILE_TRAIN_X = './data/training/'+platform+'/'+topic+'_train_x_'+str(self.pre_window)+'.csv'
    FILE_TRAIN_Y = './data/training/'+platform+'/'+topic+'_train_y_'+str(self.pre_window)+'.csv'

    train_x = []
    train_y = []
    user_list_root = []
    user_list_comm = []
    user_list = []
    num_none = 0
    with open(FILE_TRAIN_X, 'r') as f:
      for line in f:
        line = line[:-1].split(' ')
        train_x.append(line)
        for i in xrange(1, len(line)):
          if line[i] == 'userID-none':
            num_none += 1
          else:
            pass
        user_list_root.append(line[0])

    with open(FILE_TRAIN_Y, 'r') as f:
      for line in f:
        train_y.append(line[:-1])
        user_list.append(line[:-1])
        user_list_comm.append(line[:-1])


    num_none = num_none/2
    count_comm = [['UNK_COMM', -1]]
    count_comm.extend(collections.Counter(user_list_comm).most_common(self.comm_size - 1))
    count_root = [['UNK_ROOT', -1], ['userID-none', num_none]]
    count_root.extend(collections.Counter(user_list_root).most_common(self.root_size - 2))
    
    count = count_root
    count.extend(count_comm)

    dict_total = dict()
    dict_comm = dict()
    with open('./results/'+platform+'/'+topic+'/user_to_num_total.txt', 'w') as f:
      for user, _ in count:
        dict_total[user] = len(dict_total)
        f.write(user+' '+str(dict_total[user])+'\n')
    with open('./results/'+platform+'/'+topic+'/user_to_num_comm.txt', 'w') as f:
      for user, _ in count_comm:
        dict_comm[user] = len(dict_comm)
        f.write(user+' '+str(dict_comm[user])+'\n')

    unk_root = 0
    unk_comm = 0
    unk = 0
    for user in user_list_root:
      index = dict_total.get(user, 0)
      if index == 0:  # dictionary['UNK']
        unk_root += 1
    for user in user_list_comm:
      index = dict_total.get(user, 0)
      if index == 0:  # dictionary['UNK']
        unk_comm += 1
    for user in user_list:
      index = dict_comm.get(user, 0)
      if index == 0:
        unk += 1

    unk_ratio = 1.0*unk/len(user_list)
    print '  - Unknown ratio:', unk_ratio

    count[0][1] = unk_root
    count[1][1] = num_none
    count[self.root_size][1] = unk_comm


    user_root_unactive = []
    user_comm_unactive = []
    for user in user_list_root:
      if user in dict_total:
        continue
      else:
        user_root_unactive.append(user)
    for user in user_list_comm:
      if user in dict_total:
        continue
      else:
        user_comm_unactive.append(user)

    # with open('./results/'+platform+'/'+topic+'/unactive_root_user.txt', 'w') as f:
    #   for user in user_root_unactive:
    #     f.write(user+'\n')
    with open('./results/'+platform+'/'+topic+'/inactiveUserProxyFile.txt', 'w') as f:
      for user in user_comm_unactive:
        f.write(user+'\n')

    X = []
    Y = []
    n = len(train_x[0])
    for i in xrange(len(train_x)):
      x = [-1 for k in xrange(n)]
      x[0] = dict_total.get(train_x[i][0], 0)
      for j in xrange(1, n):
        x[j] = dict_total.get(train_x[i][j], self.root_size)
      X.append(x)
      Y.append(dict_comm.get(train_y[i], 0))

    prob = list()
    prob_acu = list()
    s = 0
    for word, freq in count:
      s += freq**0.75
      prob.append(freq**0.75)
      prob_acu.append(s)

    prob = np.array(prob)
    prob_acu = np.array(prob_acu)
    prob = prob/s
    self.prob_acu = prob_acu/s

    return np.array(X), np.array(Y), count, dict_total, dict_comm

  def negative_sampling(self, L_max, n):
    '''
    return n random intergers between 0 and L_max
    '''
    # x = [i for i in range(L_max)]
    # random.shuffle(x)
    # return np.array(x[0:n])
    x = list()
    for i in xrange(n):
      L = len(self.prob_acu)
      r = random.random()

      left = 0
      right = L-1

      while left<right:
        mid = int((left+right)/2)
        if self.prob_acu[mid]<r:
          left = mid + 1
        elif self.prob_acu[mid]>r:
          right = mid
        else:
          return mid

      if right!=0 and (r > self.prob_acu[right] or r < self.prob_acu[right-1]):
        print("Error!!!!!!!!", r, self.prob_acu[right], self.prob_acu[right-1])
      x.append(right)
    return x

  def plot_loss(self, path_loss):
    print 'Plot loss ...'
    loss = np.loadtxt(path_loss+'avg_loss.csv')
    plt.figure(figsize=(8,6))
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.plot(loss)
    plt.savefig(path_loss+'avg_loss.png')

if __name__ == '__main__':
  sg = skip_gram()
  Y, X, count, dict_total, dict_comm = sg.load_data(PLATFORM, TOPIC)
  sg.retrain(X, Y, learn_rate=1, Total_steps=2000000)
  path_loss = '../results/'+PLATFORM+'/'+TOPIC+'/'
  sg.plot_loss(path_loss)
#
