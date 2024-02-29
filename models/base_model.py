import pandas as pd

class BaseSelector:
    '''
    Base model class
    Defines selector model interface
    '''
    def __init__(self, **kwargs):
        '''
        Initialize model making it ready to fit
        '''
        return NotImplementedError

    def fit(data):
        '''
        fit model with data
        Arguments:
           data -- dataframe with vacancies which using to fit the model
        '''
        return NotImplementedError
        
    def predict(k: int, **kwargs):
        '''
        Gets vacancy and gives top-k sorted candidates list
        Arguments:
           vacancy -- dataframe with vacancy which model uses to find similar resumes
           k -- size of top which model returns
        '''
        return NotImplementedError
