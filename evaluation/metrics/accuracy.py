
class MRR: 
    '''
    MRR( length=20 )

    Used to iteratively calculate the average mean reciprocal rank for a result list with the defined length. 

    Parameters
    -----------
    length : int
        MRR@length
    '''
    def __init__(self, length=20):
        self.length = length;
    
    def init(self, train):
        '''
        Do initialization work here.
        
        Parameters
        --------
        train: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
        return
        
    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.test=0;
        self.pos=0
        
    def add(self, result, next_item, top_list):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        res = result[:self.length]

        if next_item in res.index:
            rank = res.index.get_loc( next_item )+1
            self.pos += ( 1.0/rank )

        self.test += 1
        
    def add_batch(self, result, next_item):
        '''
        Update the metric with a result set and the correct next item.
        
        Parameters
        --------
        result: pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.
        next_item: Array of correct next items
        '''
        i=0
        for part, series in result.iteritems(): 
            result.sort_values( part, ascending=False, inplace=True )
            self.add( series, next_item[i] )
            i += 1
        
    def result(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        return ("MRR@" + str(self.length) + ": "), (self.pos/self.test)
    
    
class HitRate: 
    '''
    MRR( length=20 )

    Used to iteratively calculate the average hit rate for a result list with the defined length. 

    Parameters
    -----------
    length : int
        HitRate@length
    '''
    
    def __init__(self, length=20):
        self.length = length;
    
    def init(self, train):
        '''
        Do initialization work here.
        
        Parameters
        --------
        train: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
        return
        
    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.test=0;
        self.hit=0
        
    def add(self, result, next_item, top_list):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        self.test += 1
        if next_item in result[:self.length].index:
            self.hit += 1

    def add_batch(self, result, next_item):
        '''
        Update the metric with a result set and the correct next item.
        
        Parameters
        --------
        result: pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.
        next_item: Array of correct next items
        '''
        i=0
        for part, series in result.iteritems(): 
            result.sort_values( part, ascending=False, inplace=True )
            self.add( series, next_item[i] )
            i += 1
        
    def result(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        return ("HitRate@" + str(self.length) + ": "), (self.hit/self.test)


class HitRateTail:
    '''
    MRR( length=20 )

    Used to iteratively calculate the average hit rate for a result list with the defined length.

    Parameters
    -----------
    length : int
        HitRate@length
    '''

    def __init__(self, length=20):
        self.length = length;

    def init(self, train):
        '''
        Do initialization work here.

        Parameters
        --------
        train: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
        return

    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.test = 0;
        self.hit = 0

    def add(self, result, next_item, top_list):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.

        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        self.test += 1
        if next_item not in top_list and next_item in result[:self.length].index:
            self.hit += 1

    def add_batch(self, result, next_item):
        '''
        Update the metric with a result set and the correct next item.

        Parameters
        --------
        result: pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.
        next_item: Array of correct next items
        '''
        i = 0
        for part, series in result.iteritems():
            result.sort_values(part, ascending=False, inplace=True)
            self.add(series, next_item[i])
            i += 1

    def result(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        return ("HitRateTail@" + str(self.length) + ": "), (self.hit / self.test)
   