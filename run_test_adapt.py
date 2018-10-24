from algorithms.knn import cknn
from algorithms.knn import iknn
from algorithms.ct import ct
import csv
import pandas as pd
import time
import scipy.io
from evaluation import evaluation as eval
from evaluation import loader as loader
from evaluation.metrics import accuracy as ac

if __name__ == '__main__':




    for dataset in ['news2']:
        for method in ['cknn']:
            '''
            Configuration
            '''
            print('---------------' + dataset + ' Adaptive  ' + method + ' --------------')
            data_path = 'data/'+dataset+'/prepared/'
            file_prefix = dataset
            limit_train = None  # limit in number of rows or None
            limit_test = None  # limit in number of rows or None
            density_value = 1  # randomly filter out events (0.0-1.0, 1:keep all)

            # create a list of metric classes to be evaluated
            metric = []
            metric.append(ac.HitRate(5))
            metric.append(ac.HitRateTail(5))
            metric.append(ac.MRR(5))
            metric.append(ac.HitRate(10))
            metric.append(ac.HitRateTail(10))
            metric.append(ac.MRR(10))
            # metric.append(cov.Coverage(20))
            # metric.append(pop.Popularity(20))

            # create a dict of (textual algorithm description => class) to be evaluated
            algs = {}

            if method == 'iknn':
            # item knn example
                knn = iknn.ItemKNN()
                algs['iknn'] = knn
            if method == 'ct':
            # item knn example
                ctt = ct.ContextTree()
                algs['CT'] = ctt
            if method == 'cknn':
                # context knn example
                knn = cknn.ContextKNN()
                algs['knn-500-1000-cosine'] = knn

            '''
            Execution
            '''
            # top_list = csv.reader('data/top_threads_' + dataset + '.csv',delimiter=',')
            # print(top_list)

            with open('data/top_threads_' + dataset + '.csv', 'r') as f:
                my_list = [list(map(int, rec)) for rec in csv.reader(f, delimiter=',')]

            top_list = my_list[0]


            all_data = pd.read_csv(data_path+dataset+'_cleaned.txt', sep='\t')

            start_time = time.time()

            rec_freshness = []
            ground_truth_freshness =[]
            correct_rec_freshness = []

            # init metrics
            for m in metric:
                m.init(all_data)
                m.reset()


            sessions = []   # sessions so far
            prev_iid = {}   # hashmap for session -> last item id
            step = 0
            session_length = {}
            for index, row in all_data.iterrows():
                step = step + 1
                train = all_data[:step]

                test = row.to_frame().transpose()

                current_session = test['SessionId'].values[0]
                current_item = test['ItemId'].values[0]


                first_in_session = False

                if current_session not in sessions: # first item in a session
                    '''
                    we don't evaluate, just fit
                    '''
                    sessions.append(current_session)
                    prev_iid[current_session] = current_item
                    first_in_session = True

                    session_length[current_session] = 1

                    # train algorithms
                    for k, a in algs.items():
                        a.fit_time_order_online(row, True)
                else:
                    '''
                    first evaluate, then fit
                    '''
                    current_session_length = session_length.get(current_session)
                    session_length[current_session] = current_session_length+1
                    # result dict
                    res = {};

                    if current_session_length < 50:
                        for k, a in algs.items():
                            res[k], rec_freshness, ground_truth_freshness, correct_rec_freshness = eval.evaluate_sessions_adapt(a, metric, test, train, prev_iid, rec_freshness, ground_truth_freshness, correct_rec_freshness, top_list, current_session_length)

                        # train algorithms
                        for k, a in algs.items():
                            a.fit_time_order_online(row, False)

                        prev_iid[current_session] = current_item

                if step % 1000 == 0:
                    print('Number of steps', step, '  Time Passed: ', time.time() - start_time, dataset, method)

                    for k, l in res.items():
                        for e in l:
                            print(k, ':', e[0], ' ', e[1])

                # print results
            with open(data_path + method + '_results_adapt.txt', 'w') as f:
                wr = csv.writer(f, dialect='excel')
                for k, l in res.items():
                    for e in l:
                        wr.writerow( [k, ':', e[0], ' ', e[1]] )

                wr.writerow(['Running Time: ', time.time()-start_time])

            scipy.io.savemat(data_path + method + '_rec_freshness', mdict={'rec_freshness': rec_freshness})
            scipy.io.savemat(data_path + method + '_ground_truth_freshness', mdict={'ground_truth_freshness': ground_truth_freshness})
            scipy.io.savemat(data_path + method + '_correct_rec_freshness', mdict={'correct_rec_freshness': correct_rec_freshness})

