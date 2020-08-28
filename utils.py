import statistics as stat


"""
This file contains some utility code, namely a logging class and a method that
extracts mean results from a training history.

authors:
    Tobia Albergoni, Quentin Bouvet, Matteo Yann Feo
"""


"""
This class prints log messages according to the current logging level.
It allows for three log levels:

- debug
- info
- warn
"""
class Logger :
    
    def __init__(self, level='info') :
        self.level(level)
        
    
    def level(self, level) :
        self.level = level
        
    
    def debug(self, msg) :
        if self.level == 'debug':
            print('[DEBG] \t' + msg)
            
            
    def info(self, msg) :
        if self.level == 'debug' or self.level == 'info':
            print('[INFO] \t' + msg)
            
    
    def warn(self, msg) :
        if self.level == 'debug' or self.level == 'info' or self.level == 'warn':
            print('[WARN] \t' + msg)
        
        
"""
This method takes a list of dictionnaries, each one containing the training history of a model,
and computes the mean of the validation accuracy, its standard deviation, and the mean training time.
"""
def validation_results(history_list):
    val_accs = [h['val_acc'][-1] for h in history_list]
    times = [h['time'][-1] for h in history_list]
    return stat.mean(val_accs), stat.stdev(val_accs), stat.mean(times)
    
    