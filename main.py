from src.data_analysis import *
from src.skip_gram import skip_gram
from src.evaluation import embedding_evaluator
import pandas as pd

PLATFORM = 'Reddit'
TOPIC = 'Cve'

N_PRE = 3

train_start = pd.to_datetime('2017-01-01').tz_localize('US/Eastern')
train_end = pd.to_datetime('2017-08-01').tz_localize('US/Eastern')
test_start = pd.to_datetime('2017-08-01').tz_localize('US/Eastern')
test_end = pd.to_datetime('2017-08-15').tz_localize('US/Eastern')


# train_start = datetime.datetime(2017,1,1,0,0,0)
# train_end = datetime.datetime(2017,8,31,23,59,59)
# test_start = datetime.datetime(2017,8,1,0,0,0)
# test_end = datetime.datetime(2017,8,14,23,59,59)

if PLATFORM == 'Reddit':
	if TOPIC == 'Crypto':
		N_ROOT = 3000
		N_COMM = 5000
		ratio_act = 0.765
		ratio_del = 0.079
		batch_size = 128
		embed_size = 128
		num_sample = 128
		K = 50
	elif TOPIC == 'Cve':
		N_ROOT = 400
		N_COMM = 5000
		ratio_act = 0.26
		ratio_del = 0.082
		batch_size = 32
		embed_size = 64
		num_sample = 32
		K = 10
	elif TOPIC == 'Cyber':
		N_ROOT = 5000
		N_COMM = 10000
		ratio_act = 0.62
		ratio_del = 0.063
		batch_size = 128
		embed_size = 128
		num_sample = 64
		K = 200
elif PLATFORM == 'Twitter':
	if TOPIC == 'Crypto':
		N_ROOT = 4000
		N_COMM = 6000
		ratio_act = 0.660
		ratio_del = 0
		batch_size = 64
		embed_size = 128
		num_sample = 64
		K = 15
	elif TOPIC == 'Cve':
		N_ROOT = 100
		N_COMM = 2000
		ratio_act = 0.980
		ratio_del = 0
		batch_size = 32
		embed_size = 128
		num_sample = 64
		K = 5
	elif TOPIC == 'CveS2':
		N_ROOT = 93
		N_COMM = 375
		ratio_act = 1.0
		ratio_del = 0
		batch_size = 32
		embed_size = 128
		num_sample = 64
		K = 5
	elif TOPIC == 'Cyber':
		N_ROOT = 20000
		N_COMM = 30000
		ratio_act = 0.339
		ratio_del = 0.0
		batch_size = 128
		embed_size = 128
		num_sample = 64
		K = 20

start = datetime.datetime.now()
if PLATFORM == 'Reddit':
	get_post_cascade(PLATFORM, TOPIC, train_start, train_end)
	get_post_cascade(PLATFORM, TOPIC, test_start, test_end)
elif PLATFORM == 'Twitter':
	get_post_cascade_twitter(PLATFORM, TOPIC, train_start, train_end)
	get_post_cascade_twitter(PLATFORM, TOPIC, test_start, test_end)
get_train_data(PLATFORM, TOPIC, train_start, train_end, N_PRE)
get_parent_prob(PLATFORM, TOPIC, train_start, train_end)
get_all_user_list(PLATFORM, TOPIC, train_start, train_end)
get_user_post_scale(PLATFORM, TOPIC, train_start, train_end)
analysis_cascade(PLATFORM, TOPIC, N_ROOT, N_COMM, train_start, train_end)

params_file = './src/params/'+PLATFORM+'/'+TOPIC+'/params.txt'
sg = skip_gram(N_ROOT, N_COMM, batch_size, embed_size, N_PRE, num_sample, params_file)
Y, X, count, dict_total, dict_comm = sg.load_data(PLATFORM, TOPIC)
sg.train(X, Y, PLATFORM, TOPIC, learn_rate=1, Total_steps=500000)
path_loss = './results/'+PLATFORM+'/'+TOPIC+'/'
sg.plot_loss(path_loss)

PATH_SCORE = './results/'+PLATFORM+'/'+TOPIC+'/scoreMatrixProxyFile.txt'
GRDTRU = './data/cascade/'+PLATFORM+'/cascade_'+TOPIC+'_'+str(train_start.date())+'---'+str(train_end.date())+'.txt'
TEST = './data/cascade/'+PLATFORM+'/cascade_'+TOPIC+'_'+str(test_start.date())+'---'+str(test_end.date())+'.txt'

eva = embedding_evaluator(n_pre=N_PRE, k=K, path_score=PATH_SCORE, N_ROOT=N_ROOT, N_COMM=N_COMM)
print 'Generating score matrix ...'
eva.generate_score_matrix(PLATFORM, TOPIC)
print 'Loading score matrix ...'
eva.load_score()
eva.analysis(PLATFORM, TOPIC, GRDTRU, TEST, ratio_act, ratio_del)
# eva.compare_init_true(PLATFORM, TOPIC, GRDTRU, TEST)
print 'Running time:', datetime.datetime.now()-start
