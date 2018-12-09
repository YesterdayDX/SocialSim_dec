import pandas as pd
import numpy as np
import datetime

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import collections
from decimal import *

COLUMNS = ['nodeID', 'nodeUserID', 'parentID', 'rootID', 'actionType', 'nodeTime']
# train_start = datetime.datetime(2017,1,1,0,0,0)
# train_end = datetime.datetime(2017,8,31,23,59,59)
# test_start = datetime.datetime(2017,8,1,0,0,0)
# test_end = datetime.datetime(2017,8,14,23,59,59)


NONE = 'userID-none'

def get_post_cascade(platform, topic, time_start, time_end):
    '''
    From event file, getting posts.
    Each row is a post, starting with rootID, followed by the user who publish the post and make comments.
    rootID: root-rootUserID user1 user2 user3 ...
    '''
    print 'Get post cascade of '+platform+'_'+topic+': '+str(time_start.date()) + '--' + str(time_end.date())

    file_event = './data/events/'+platform+'/'+topic+'_event.csv'
    print "  - Event file: " + file_event
    df = pd.read_csv(file_event, header=0)
    df = df.drop_duplicates()
    df['nodeTime'] = pd.to_datetime(df['nodeTime'])
    df = df.sort_values(by='nodeTime')
    df = df[df['nodeTime']>=time_start]
    df = df[df['nodeTime']<=time_end]
    # print df

    df_g = df.groupby(['rootID'])
    dict_cascade = dict()
    dict_root_to_user = dict()

    for rootID, post in df_g:
        if rootID in dict_cascade:
            print "Error: Duplicated rootID!"
            return

        cascade = list()
        pc = post.groupby(['actionType'])
        if len(pc) == 1:
            for actionType, dfg in pc:
                if actionType == 'post':
                    dict_cascade[rootID] = ['root-'+dfg['nodeUserID'].iloc[0]]
                else:
                    #print dfg
                    for i in dfg.index:
                        cascade.append(dfg.loc[i, 'nodeUserID'])
                    dict_cascade[rootID] = list()
                    dict_cascade[rootID].extend(cascade)
        else:
            for actionType, dfg in pc:
                if actionType == 'post':
                    dict_cascade[rootID] = list()
                    dict_cascade[rootID].append('root-'+dfg['nodeUserID'].iloc[0])
                elif actionType == 'comment':
                    for i in dfg.index:
                        cascade.append(dfg.loc[i, 'nodeUserID'])
            dict_cascade[rootID].extend(cascade)
    print '  - Number of posts:', len(dict_cascade)

    file_cas = './data/cascade/'+platform+'/cascade_'+topic+'_'+str(time_start.date())+'---'+str(time_end.date())+'.txt'
    print "  - Write cascade to " + file_cas
    with open(file_cas, 'w') as f:
        for rootID in dict_cascade:
            if len(dict_cascade)<=1:
                continue
            f.write(rootID+': ')
            f.write(dict_cascade[rootID][0])
            for i in range(1, len(dict_cascade[rootID])):
                f.write(' ' + dict_cascade[rootID][i])
            f.write('\n')

def get_post_cascade_twitter(platform, topic, time_start, time_end):
    '''
    From event file, getting posts.
    Each row is a post, starting with rootID, followed by the user who publish the post and make comments.
    rootID: root-rootUserID user1 user2 user3 ...
    '''
    print 'Get post cascade of '+platform+'_'+topic+': '+str(time_start.date()) + '--' + str(time_end.date())
    df = pd.read_csv('./data/events/'+platform+'/'+topic+'_event.csv', header=0)
    df = df.drop_duplicates()

    df['nodeTime'] = pd.to_datetime(df['nodeTime'])
    df = df.sort_values(by='nodeTime')
    df = df[df['nodeTime']>=time_start]
    df = df[df['nodeTime']<=time_end]

    df_g = df.groupby(['rootID'])
    dict_cascade = dict()
    dict_root_to_user = dict()

    x = 0
    for rootID, post in df_g:
        if rootID in dict_cascade:
            print "Error: Duplicated rootID!"
            return

        cascade = list()
        pc = post.groupby(['actionType'])

        if len(pc) == 1:
            for actionType, dfg in pc:
                if actionType == 'tweet':
                    dict_cascade[rootID] = ['root-'+dfg['nodeUserID'].iloc[0]]
                else:
                    #print dfg
                    for i in dfg.index:
                        cascade.append(dfg.loc[i, 'nodeUserID'])
                    dict_cascade[rootID] = list()
                    dict_cascade[rootID].extend(cascade)
        else:
            c = 0
            for actionType, dfg in pc:
                if actionType == 'tweet':
                    c = 1
                    dict_cascade[rootID] = list()
                    dict_cascade[rootID].append('root-'+dfg['nodeUserID'].iloc[0])
                else:
                    for i in dfg.index:
                        cascade.append(dfg.loc[i, 'nodeUserID'])
            if c==1:
                dict_cascade[rootID].extend(cascade)
            else:
                # print '# wrong:', x
                x += 1
    print '  - Number of posts:', len(dict_cascade)

    with open('./data/cascade/'+platform+'/cascade_'+topic+'_'+str(time_start.date())+'---'+str(time_end.date())+'.txt', 'w') as f:
        for rootID in dict_cascade:
            if len(dict_cascade)<=1:
                continue
            f.write(rootID+': ')
            f.write(dict_cascade[rootID][0])
            for i in range(1, len(dict_cascade[rootID])):
                f.write(' ' + dict_cascade[rootID][i])
            f.write('\n')

def get_train_data(platform, topic, time_start, time_end, n_pre):
    """
    Get training data from several list of cascades
    rootID: rootuser u1 u2 u3 u4 ...
    n_pre: number of users used to predict the next one (the rootuser is always included)
    """
    print 'Get training data ...'
    file_cascade = './data/cascade/'+platform+'/cascade_'+topic+'_'+str(time_start.date())+'---'+str(time_end.date())+'.txt'
    dict_cascade = dict()
    with open(file_cascade, 'r') as f:
        for line in f:
            line = line[:-1].split(' ')
            dict_cascade[line[0][:-1]] = line[1:]

    train_x = []
    train_y = []

    for rootID in dict_cascade:
        cas = dict_cascade[rootID]
        if len(cas) == 1:
            continue
        if cas[0][:5] != 'root-':
            continue
        for i in xrange(1, min(len(cas), n_pre+1)):
            y = cas[i]
            x = [cas[0]]
            for j in xrange(n_pre-i+1):
                x.append(NONE)
            for j in xrange(1, i):
                x.append(cas[j])

            train_x.append(x)
            train_y.append(y)

        for i in range(n_pre+1, len(cas)):
            x = [cas[0]]
            x.extend(cas[i-n_pre:i])
            train_x.append(x)
            train_y.append(cas[i])

    fx = open('./data/training/'+platform+'/'+topic+'_train_x_'+str(n_pre)+'.csv', 'w')
    fy = open('./data/training/'+platform+'/'+topic+'_train_y_'+str(n_pre)+'.csv', 'w')
    for i in xrange(len(train_x)):
        fx.write(' '.join(train_x[i])+'\n')
        fy.write(train_y[i]+'\n')

    fx.close()
    fy.close()

def get_parent_prob(platform, topic, time_start, time_end, N=5):
    '''
    Read in event file.
    Get the top N(=5) parents and their corresponding probabilities.
    '''
    print 'Get parent probability ...'
    file_event = './data/events/'+platform+'/'+topic+'_event.csv'

    df = pd.read_csv(file_event, index_col=0, header=0)
    df = df.drop_duplicates()
    df['nodeTime'] = pd.to_datetime(df['nodeTime'])
    df = df.sort_values(by='nodeTime')
    df = df[df['nodeTime']>=time_start]
    df = df[df['nodeTime']<=time_end]

    parent = dict()
    allnodes = df.index.values

    dict_node = dict()
    for i in xrange(df.shape[0]):
        dict_node[allnodes[i]] = i
    X = df.values

    for i in xrange(df.shape[0]):
        node = X[i]
        nodeUserID = node[0]
        parentID = node[1]

        if parentID not in dict_node:
            continue
        else:
            parentUserID = X[dict_node[parentID]][0]

        if nodeUserID in parent:
            parent[nodeUserID].append(parentUserID)
        else:
            parent[nodeUserID] = [parentUserID]

    N = 5
    # print 'Number of users: '+str(len(parent))
    for nodeUserID in parent:
        parent[nodeUserID] = collections.Counter(parent[nodeUserID]).most_common(N)

    # Get the ratio of following root or not
    num_root = 0
    num_total = 0
    num_not_root = 0

    for i in xrange(df.shape[0]):
        if X[i][1] != 'nan':
            if X[i][1] == X[i][2]:
                num_root += 1
            else:
                num_not_root += 1
            num_total += 1

    print '  - Parent is root:', 1.0*num_root/num_total

    with open('./results/'+platform+'/'+topic+'/commentProbabilityProxyFile.txt', 'w') as f:
        for nodeUserID in sorted(set(df['nodeUserID'].values)):
            f.write(nodeUserID+' ')

            if nodeUserID in parent:
                S = 0
                for parentUserID, num in parent[nodeUserID]:
                    S += num
                for parentUserID, num in parent[nodeUserID]:
                    f.write(parentUserID+' '+str(1.0*num/S)+' ')
                f.write('\n')

            else:
                f.write('[root] 1.0\n')

def get_all_user_list(platform, topic, time_start, time_end):
    file_event = './data/events/'+platform+'/'+topic+'_event.csv'
    print 'Get user list ...'

    df = pd.read_csv(file_event, index_col=0, header=0)
    df = df.drop_duplicates()
    df['nodeTime'] = pd.to_datetime(df['nodeTime'])
    df = df[df['nodeTime']>=time_start]
    df = df[df['nodeTime']<=time_end]

    file_userID = './results/'+platform+'/'+topic+'/userID.txt'

    with open(file_userID, 'w') as f:
        for ID in sorted(set(df['nodeUserID'].values)):
            f.write(ID+'\n')

def get_user_post_scale(platform, topic, time_start, time_end):
    file_cascade = './data/cascade/'+platform+'/cascade_'+topic+'_'+str(time_start.date())+'---'+str(time_end.date())+'.txt'

    print 'Get user post scale .. '
    dict_cascade = dict()
    user_post_scale = dict()
    with open(file_cascade, 'r') as f:
        for line in f:
            line = line[:-1].split(' ')

            if line[1][:5] == 'root-':
                rootUser = line[1][5:]
            else:
                continue

            if rootUser not in user_post_scale:
                user_post_scale[rootUser] = [len(line)-2]
            else:
                user_post_scale[rootUser].append(len(line)-2)

    user_list = []
    file_userID = './results/'+platform+'/'+topic+'/userID.txt'
    with open(file_userID, 'r') as f:
        for line in f:
            user_list.append(line[:-1])

    file_scale = './results/'+platform+'/'+topic+'/user_post_scale.txt'
    with open(file_scale, 'w') as f:
        for user in user_list:
            x = user_post_scale.get(user, [0.0])
            f.write(user+' '+str(np.mean(x))+' '+str(np.std(x))+'\n')


def merge_post_num_scale(platform, topic, time_start, time_end, scenario=1):
    '''
    Used to merge the post scale with RNN or ARIMA
    '''
    n = (time_end - time_start).days + 1
    if scenario==1:
        file_ori = '../data/postScale/postScaleProxyFile_'+platform+'_'+topic+'.txt'
        file_sca = '../results/'+platform+'/'+topic+'/user_post_scale.txt'

        df_ori = pd.read_csv(file_ori, header=None, index_col=0, sep = ' ')
        df_ori = df_ori.sort_index()

        df_sca = pd.read_csv(file_sca, header=None, index_col=0, sep = ' ')
        df_sca = df_sca.sort_index()

        if df_ori.shape[0] != df_sca.shape[0]:
            print 'Error: Unequal shape'
            return
        for i in xrange(df_ori.shape[0]):
            if df_ori.index.values[i] != df_sca.index.values[i]:
                print 'Error: different users!'
                return

        X = df_ori.values

        MAX = (df_sca.values[:,0]+df_sca.values[:,1]).reshape((df_sca.shape[0], 1))
        scale = np.random.random([df_ori.shape[0], df_ori.shape[1]/2]) * MAX

        for i in xrange(scale.shape[0]):
            if df_sca.values[i][1] == 0:
                mu = df_sca.values[i][0]
                for j in xrange(scale.shape[1]):
                    scale[i][j] = mu

        for i in xrange(1, X.shape[1], 2):
            X[:,i] = scale[:,(i-1)/2]

        df = pd.DataFrame(X, index=df_ori.index)

        file_merge = '../results/'+platform+'/'+topic+'/postScaleProxyFile.txt'
        df.to_csv(file_merge, header=None, sep = ' ')

    elif scenario==2:
        file_sca = '../results/'+platform+'/'+topic+'/user_post_scale.txt'
        df_sca = pd.read_csv(file_sca, header=None, index_col=0, sep = ' ')
        df_sca = df_sca.sort_index()
        n = (time_end-time_start).days + 1
        print n, 'days'

        X = -np.ones((df_sca.shape[0], n))

        MAX = (df_sca.values[:,0]+df_sca.values[:,1]).reshape((df_sca.shape[0], 1))
        scale = np.random.random([df_sca.shape[0], n]) * MAX

        for i in xrange(scale.shape[0]):
            if df_sca.values[i][1] == 0:
                mu = df_sca.values[i][0]
                for j in xrange(scale.shape[1]):
                    scale[i][j] = mu

        for i in xrange(1, X.shape[1], 2):
            X[:,i] = scale[:,(i-1)/2]

        df = pd.DataFrame(X, index=df_sca.index)
        print df.shape

        file_merge = '../results/'+platform+'/'+topic+'/postScaleProxyFile.txt.s2'
        df.to_csv(file_merge, header=None, sep = ' ')

        # print np.random.random([4,2])*np.array([[1],[10],[100],[1000]])
        # print df_sca


def analysis_cascade(platform, topic, n_root, n_comm, time_start, time_end):
    file_cascade = './data/cascade/'+platform+'/cascade_'+topic+'_'+str(time_start.date())+'---'+str(time_end.date())+'.txt'

    print "Analysis cascade ..."
    dict_cascade = dict()
    cascade_max_len = 0
    with open(file_cascade, 'r') as f:
        for line in f:
            line = line[:-1].split(' ')
            dict_cascade[line[0][:-1]] = line[1:]
            if cascade_max_len < len(line)-1:
                cascade_max_len = len(line)-1

    # Calculate the distribution of cascade length
    len_list = np.zeros(1+cascade_max_len, dtype='int32')
    root_user_list = []
    comm_user_list = []

    for postID in dict_cascade:
        len_list[len(dict_cascade[postID])] += 1
        post = dict_cascade[postID]
        root_user_list.append(post[0])
        comm_user_list.extend(post[1:])

    f = open('./results/'+platform+'/'+topic+'/log.txt', 'w')
    f.write(str(time_start.date())+'---'+str(time_end.date())+'\n')
    f.write("Maximum cascade length: "+str(cascade_max_len)+'\n')
    f.write("Number of cascade: "+str(len_list.sum())+'\n')

    len_list_draw = len_list[:20]
    f.write("# of cascade (L<20): "+str(len_list_draw.sum())+'\n')
    f.write("Percentage of cascade (L<20): "+str(1.0*len_list_draw.sum()/len_list.sum())+'\n')

    plt.figure(figsize=(8,6))
    plt.xlabel("Length of cascade")
    plt.ylabel("Percentage (%)")
    plt.bar(range(len(len_list_draw)), 100.0*len_list_draw/len_list.sum())
    plt.savefig('./results/'+platform+'/'+topic+'/Dist_cas_len.png')

    root_count = collections.Counter(root_user_list)
    x = sorted(root_count.values(), reverse=1)
    plt.figure(figsize=(8,6))
    plt.xlabel("Top 10000 users")
    plt.ylabel("# of actions")
    plt.plot(x[1:n_root])
    plt.savefig('./results/'+platform+'/'+topic+'/root_user_act.png')
    f.write('\n============= Root User ================\n')
    f.write('Number of root user: '+str(len(x))+'\n')
    f.write('Mean: %.2f, Median: %.2f, X[%d]: %d\n' %(np.mean(x), np.median(x), n_root, x[n_root]))
    f.write('Ratio top %d : %.2f\n' %(n_root, 1.0*np.sum(x[0:n_root])/np.sum(x)))

    comm_count = collections.Counter(comm_user_list)
    d = comm_count['[Deleted]']
    x = sorted(comm_count.values(), reverse=1)
    plt.figure(figsize=(8,6))
    plt.xlabel("Top 10000 users")
    plt.ylabel("# of actions")
    plt.plot(x[1:n_comm])
    plt.savefig('./results/'+platform+'/'+topic+'/comm_user_act.png')
    f.write('\n============== Comment User ================\n')
    f.write('Number of comment user: '+str(len(x))+'\n')
    f.write('Mean: %.2f, Median: %.2f, X[%d]: %d\n' %(np.mean(x), np.median(x), n_comm, x[n_comm]))
    f.write('Ratio top %d : %.2f\n' %(n_comm, 1.0*np.sum(x[0:n_comm])/np.sum(x)))
    f.write('Deleted ratio: '+str(1.0*d/np.sum(x))+'\n')

if __name__ == '__main__':    
    n_pre = 3
    n_root = N_ROOT
    n_comm = N_COMM
    start = datetime.datetime.now()
    # get_post_cascade(PLATFORM, TOPIC, train_start, train_end)
    # get_train_data(PLATFORM, TOPIC, train_start, train_end, n_pre)
    # get_parent_prob(PLATFORM, TOPIC, train_start, train_end)
    # analysis_cascade(PLATFORM, TOPIC, n_root, n_comm, train_start, train_end)
    # get_user_post_scale(PLATFORM, TOPIC, train_start, train_end)
    # get_all_user_list(PLATFORM, TOPIC, train_start, train_end)
    # merge_post_num_scale(PLATFORM, TOPIC, test_start, test_end, 1)
    print 'Running time:', datetime.datetime.now()-start


