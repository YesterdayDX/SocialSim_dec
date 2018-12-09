import numpy as np
import pandas as pd
from random import choice
import matplotlib.pyplot as plt
from datetime import datetime

# N_PRE = 3

WEIGHT = [0.25, 0.25, 0.25, 0.25]
# PLATFORM = 'Reddit'
# TOPIC = 'Cyber'

# PATH_SCORE1 = '../results/score_1.csv'
# PATH_SCORE2 = '../results/'+PLATFORM+'/'+TOPIC+'/score.csv'

# if TOPIC == 'Crypto':
# 	N_ROOT = 3000
# 	N_COMM = 5000
# 	ratio_act = 0.765
# 	ratio_del = 0.079
# 	K = 50
# elif TOPIC == 'Cve':
# 	N_ROOT = 500
# 	N_COMM = 3000
# 	ratio_act = 0.531
# 	ratio_del = 0.082
# 	K = 10
# elif TOPIC == 'Cyber':
# 	N_ROOT = 5000
# 	N_COMM = 20000
# 	ratio_act = 0.535
# 	ratio_del = 0.064
# 	K = 200

# GRDTRU = '../data/cascade/'+PLATFORM+'/cascade_'+TOPIC+'_2017-08-01---2017-08-14.txt'
# INIT = '../data/cascade/'+PLATFORM+'/cascade_'+TOPIC+'_2017-01-01---2017-08-31.txt'

# ratio_act = 0.760
# ratio_del = 0.075


class embedding_evaluator:
	def __init__(self, n_pre, k, path_score, N_ROOT, N_COMM, weight=WEIGHT):
		self.n_pre = n_pre
		self.k = k
		self.weight = weight
		self.path_score = path_score
		self.score = -1
		self.num_to_user_row = dict()
		self.num_to_user_col = dict()
		self.user_to_num_row = dict()
		self.user_to_num_col = dict()

		self.n_root = N_ROOT
		self.n_comm = N_COMM

		self.cas_tru = 0
		self.cas_gen = 0
		self.cas_ini = 0

	def generate_score_matrix(self, PLATFORM, TOPIC):
		embeddings_in = np.loadtxt('./results/'+PLATFORM+'/'+TOPIC+'/embeddings_in.csv')
		embeddings_out = np.loadtxt('./results/'+PLATFORM+'/'+TOPIC+'/embeddings_out.csv')
		# print 'Embedding_in shape:', embeddings_in.shape
		# print 'Embedding_out shape:', embeddings_out.shape

		#######################
		norm = np.sqrt(np.sum(np.square(embeddings_out), 1, keepdims=True))
  		embeddings_out = embeddings_out / norm
  		norm = np.sqrt(np.sum(np.square(embeddings_in), 1, keepdims=True))
  		embeddings_in = embeddings_in / norm
  		########################
		
		user_to_num_total = dict()
		user_to_num_comm = dict()
		with open('./results/'+PLATFORM+'/'+TOPIC+'/user_to_num_total.txt', 'r') as f:
			for line in f:
				line = line[:-1].split(' ')
				user_to_num_total[line[0]] = int(line[1])
		with open('./results/'+PLATFORM+'/'+TOPIC+'/user_to_num_comm.txt', 'r') as f:
			for line in f:
				line = line[:-1].split(' ')
				user_to_num_comm[line[0]] = int(line[1])

		num_to_user_total = dict(zip(user_to_num_total.values(), user_to_num_total.keys()))
		num_to_user_comm = dict(zip(user_to_num_comm.values(), user_to_num_comm.keys()))

		SCORE = np.dot(embeddings_out, embeddings_in.T)
		index = [num_to_user_total[i] for i in range(len(num_to_user_total))]
		column = [num_to_user_comm[i] for i in range(len(num_to_user_comm))]

		df = pd.DataFrame(SCORE, index=index, columns=column)
		df = df.iloc[:,2:]
		df.to_csv(self.path_score)
		# print "Wrote score matrix to file"

	def load_score(self):
		df = pd.read_csv(self.path_score, index_col=0)
		'Read score matrix completed!'
		self.score = df.values

		for i in xrange(len(df.index.values)):
			self.user_to_num_row[df.index[i]] = i
		for i in xrange(len(df.columns.values)):
			self.user_to_num_col[df.columns[i]] = i
		self.num_to_user_row = dict(zip(self.user_to_num_row.values(), self.user_to_num_row.keys()))
		self.num_to_user_col = dict(zip(self.user_to_num_col.values(), self.user_to_num_col.keys()))

	def get_top_k_similar(self, root, pres, k):
		index = [root]
		index.extend(pres)
		subscore = self.score[index,:]
		subscore = np.dot(subscore.T, self.weight)
		topk = np.argsort(-subscore)[0: k]
		return topk

	def generate_post(self, rootID, L, ratio_act, ratio_del):
		"""
		Generate a post
		Return: [RootID, UserId, UserId, UserId, ...]
		"""
		root = self.user_to_num_row.get(rootID, 0)
		post = [root, 1, 1, 1]
		post_user = [rootID]
		for i in xrange(1, L):
			r = np.random.random()
			# r = 0
			if r>ratio_act:
				post.append(self.n_root)
				post_user.append('UNK_COMM')
			else:
				r = np.random.random()
				# r = 1
				if r<ratio_del:
					post.append(self.n_root+1)
					post_user.append('[Deleted]')
				else:
					topk = self.get_top_k_similar(post[0], post[-self.n_pre:], self.k)
					target = choice(topk)
					target_user = self.num_to_user_col[target]

					post.append(self.user_to_num_row[target_user])
					post_user.append(target_user)

		post = post[:1] + post[4:]
		return post, post_user

	def load_ground_truth(self, path_grdtru, path_init):
		cascade = []
		with open(path_grdtru, 'r') as f:
			for line in f:
				line = line[:-1].split(' ')
				line = line[1:]
				if len(line)<=1:
					continue
				if line[0][0:5] == 'root-':
					# line[0] = line[0][5:]
					cascade.append(line)
				else:
					# print "Error: lacking root!"
					continue
		self.cas_tru = cascade

		cascade = []
		with open(path_init, 'r') as f:
			for line in f:
				line = line[:-1].split(' ')
				line = line[1:]
				if len(line)<=1:
					continue
				if line[0][0:5] == 'root-':
					# line[0] = line[0][5:]
					cascade.append(line)
				else:
					# print "Error: lacking root!"
					continue
		self.cas_ini = cascade

	def compare_init_true(self, PLATFORM, TOPIC, GRDTRU, TEST):
		print 'Comparing init and test ...'
		self.load_score()
		self.load_ground_truth(GRDTRU, TEST)

		dist_tru = [0 for i in xrange(self.n_comm)]
		dist_ini = [0 for i in xrange(self.n_comm)]

		for post in self.cas_tru:
			for ID in post[1:]:
				dist_tru[self.user_to_num_col.get(ID, 0)] += 1
		for post in self.cas_ini:
			for ID in post[1:]:
				dist_ini[self.user_to_num_col.get(ID, 0)] += 1

		np.savetxt('./results/'+PLATFORM+'/'+TOPIC+'/dist_ini.csv', dist_ini)
		np.savetxt('./results/'+PLATFORM+'/'+TOPIC+'/dist_tru.csv', dist_tru)

		print '  - Max ini: %d, arg: %d' %(max(dist_ini), np.argmax(dist_ini))
		print '  - Max tru: %d, arg: %d' %(max(dist_tru), np.argmax(dist_tru))
		print '  - Ini: unknown=%d, delete=%d' %(dist_ini[0], dist_ini[1])
		print '  - Tru: unknown=%d, delete=%d' %(dist_tru[0], dist_tru[1])
		print '  - Sum:', sum(dist_ini), sum(dist_tru)

		dist_tru = dist_tru[2:]
		dist_ini = dist_ini[2:]

		# print dist_gen
		plt.figure(figsize=(15,6))
		plt.xlabel("User")
		plt.ylabel("# of actions")
		fig,axes=plt.subplots(2,1,figsize=(15,6))
		axes[0].set_title('Ground Truth')
		axes[0].plot(range(len(dist_tru[2:])), dist_tru[2:])
		axes[1].set_title('Generated Post')
		axes[1].plot(range(len(dist_ini[2:])), dist_ini[2:])
		plt.savefig('./results/'+PLATFORM+'/'+TOPIC+'/tru&ini.png')

	def analysis(self, PLATFORM, TOPIC, GRDTRU, TEST, ratio_act, ratio_del):
		print 'Analysis '+PLATFORM+' '+TOPIC + ' ...'
		# print 'Ground Truth: '+GRDTRU
		self.load_ground_truth(GRDTRU, TEST)
		cas_gen = []
		for post in self.cas_tru:
			rootID = post[0]
			_, post_gen = self.generate_post(rootID, len(post), ratio_act, ratio_del)
			cas_gen.append(post_gen)


		dist_tru = [0 for i in xrange(self.n_comm)]
		dist_gen = [0 for i in xrange(self.n_comm)]

		for post in self.cas_tru:
			for ID in post[1:]:
				dist_tru[self.user_to_num_col.get(ID, 0)] += 1
		for post in cas_gen:
			for ID in post[1:]:
				if ID == 'UNK_COMM':
					dist_gen[0] += 1
				elif ID == '[Deleted]':
					dist_gen[1] += 1
				else:
					dist_gen[self.user_to_num_col[ID]] += 1

		np.savetxt('./results/'+PLATFORM+'/'+TOPIC+'/dist_gen.csv', dist_gen)
		np.savetxt('./results/'+PLATFORM+'/'+TOPIC+'/dist_tru.csv', dist_tru)

		print '  - Max gen: %d, arg: %d' %(max(dist_gen), np.argmax(dist_gen))
		print '  - Max tru: %d, arg: %d' %(max(dist_tru), np.argmax(dist_tru))
		print '  - Gen: unknown=%d, delete=%d' %(dist_gen[0], dist_gen[1])
		print '  - Tru: unknown=%d, delete=%d' %(dist_tru[0], dist_tru[1])
		print '  - Sum:', sum(dist_gen), sum(dist_tru)

		dist_tru = dist_tru[2:]
		dist_gen = dist_gen[2:]

		# print dist_gen
		plt.figure(figsize=(15,6))
		plt.xlabel("User")
		plt.ylabel("# of actions")
		fig,axes=plt.subplots(2,1,figsize=(15,6))
		axes[0].set_title('Ground Truth')
		axes[0].plot(range(len(dist_tru[2:])), dist_tru[2:])
		axes[1].set_title('Generated Post')
		axes[1].plot(range(len(dist_gen[2:])), dist_gen[2:])
		plt.savefig('./results/'+PLATFORM+'/'+TOPIC+'/tru&gen.png')

	def draw(self):
		dist_gen = np.loadtxt('./results/'+PLATFORM+'/'+TOPIC+'/dist_gen.csv')
		dist_tru = np.loadtxt('./results/'+PLATFORM+'/'+TOPIC+'/dist_tru.csv')
		plt.figure(figsize=(15,6))
		plt.xlabel("User")
		plt.ylabel("# of actions")
		fig,axes=plt.subplots(2,1,figsize=(15,6))
		axes[0].set_title('Ground Truth')
		axes[0].plot(range(len(dist_tru[2:])), dist_tru[2:])
		axes[1].set_title('Generated Post')
		axes[1].plot(range(len(dist_gen[2:])), dist_gen[2:])
		plt.savefig('./results/'+PLATFORM+'/'+TOPIC+'/tru&gen.png')



if __name__ == '__main__':
	start = datetime.now()
	eva = embedding_evaluator()
	eva.generate_score_matrix()
	eva.load_score()
	eva.analysis()
	# eva.compare_init_true()
	print 'Running time:', datetime.now()-start
	# eva.draw()
