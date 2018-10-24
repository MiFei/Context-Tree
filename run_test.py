
from algorithms.knn import cknn
#from algorithms.smf import  smf
from algorithms.knn import scknn
from algorithms.ct import ct
from algorithms.knn import iknn
#from algorithms.gru4rec import gru4rec
from evaluation import evaluation as eval
from evaluation import loader as loader
from evaluation.metrics import accuracy as ac
import time
import csv
import os
import psutil
import pickle
import gc

if __name__ == '__main__':
    
    '''
    Configuration
    '''

    for dataset in ['rsc15', 'course3', 'news2']:
        for method in ['ct']:
            print('---------------'+dataset+ ' Static  ' + method + ' --------------')
            data_path = 'data/' + dataset + '/prepared/'
            file_prefix = dataset
            limit_train = None #limit in number of rows or None
            limit_test = None #limit in number of rows or None
            density_value = 1 #randomly filter out events (0.0-1.0, 1:keep all)

            # create a list of metric classes to be evaluated
            metric = []
            metric.append( ac.HitRate(20) )
            metric.append( ac.HitRate(5) )
            metric.append( ac.MRR(20) )
            metric.append( ac.MRR(5) )

            # create a dict of (textual algorithm description => class) to be evaluated
            algs = {}

            isCT = False
            isHybrid = False
            if method == 'gru4rec':
                gru = gru4rec.GRU4Rec( layers = [1000], loss='top1', final_act='tanh', hidden_act='tanh', dropout_p_hidden=0.1, learning_rate=0.05, momentum=0.1 )
                algs['gru-100-top1'] = gru
            if method == 'gru4rec2':
                gru = gru4rec.GRU4Rec( loss='bpr-max-0.5', final_act='tanh', hidden_act='tanh', layers=[1000], dropout_p_hidden=0.1, learning_rate=0.05, momentum=0.1, n_sample=2048, sample_alpha=0)
                algs['gru-100-bpr-max-0.5'] = gru
            if method == 'iknn':
            # item knn example
                knn = iknn.ItemKNN()
                algs['iknn'] = knn
            if method == 'ct':
            # item knn example
                ctt = ct.ContextTree()
                algs['CT'] = ctt
                isCT = True
            if method == 'cknn':
                # context knn example
                knn = cknn.ContextKNN()
                algs['knn-500-1000-cosine'] = knn
            if method == 'scknn':
                scknn_obj = scknn.SeqContextKNN(100, 500, similarity="cosine", extend=False)
                algs['scknn-100-500-cosine-div'] = scknn_obj
            if method == 'smf':
                smf_obj = smf.SessionMF(factors=100, batch=50, learn='adagrad_sub', learning_rate=0.085, momentum=0.2,
                                    regularization=0.005, dropout=0.3, skip=0.0, epochs=10, shuffle=-1,
                                    activation='linear', objective='top1_max', samples=2048)
                algs['smf-bpr'] = smf_obj
            '''
            Execution
            '''
            #load data
            train, test = loader.load_data( data_path, file_prefix, rows_train=limit_train, rows_test=limit_test, density=density_value)
            item_ids = train.ItemId.unique()


            #init metrics
            for m in metric:
                m.init( train )

            #train algorithms
            start_time_training = time.time()
            for k,a in algs.items():
                a.fit( train )

            end_time_training = time.time()
            #result dict
            res = {};

            print('Total Traing Time: ', end_time_training - start_time_training)

            start_time_testing = time.time()
            #evaluation
            for k, a in algs.items():
                res[k], time_sum, time_count = eval.evaluate_sessions( a, metric, test, train, isHybrid, isCT)
            end_time_testing = time.time()


            #print results
            with open(data_path + method + '_results_static.txt', 'w') as f:
                wr = csv.writer(f, dialect='excel')
                for k, l in res.items():
                    for e in l:
                        wr.writerow( [k, ':', e[0], ' ', e[1]] )
                        print( k, ':', e[0], ' ', e[1] )

                wr.writerow(['Total Traing Time: ', end_time_training - start_time_training])
                wr.writerow(['Testing Time sum: ', time_sum, 'Time count: ', time_count, 'Avg Response Time:', time_sum/time_count])
