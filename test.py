from training_framework import TrainingFramework
from architectures import ShallowFullyConnectedArchitecture, DeepFullyConnectedArchitecture, ConvolutionalArchitecture
from utils import Logger, validation_results


"""
This file can be executed to reproduce the work and results reported on the report of the first miniproject.
The main method trains all three architectures with the four variants, thus producing 12 training histories.
Each training run is executed with freshly loaded data and with random initialization of model parameters.

Architectures:
- Shallow fully connected net
- Deep fully connected net
- Convolutional net

Variants:
- No weight sharing, no auxiliary task
- Weight sharing, no auxiliary task
- No weight sharing, auxiliary task
- Weight sharing, auxiliary task

The code prints log messages showing the status and results of each training run, and in the end prints a summary
of the average results achieved by each model. Note that the code in this file takes approximately 40 minutes to run
on a machine with 8GB of memory and a 2.5 GHz CPU.

The bulk of the code and the relative documentation can be found in the training_framework.py file.

authors:
    Tobia Albergoni, Quentin Bouvet, Matteo Yann Feo
"""


def main():
    # Create learning framework
    trn_framework = TrainingFramework(log_level='warn')
    
    # Validate the four variants of the shallow fully connected architecture
    trn_framework.enable_aux_task(False)
    history_FCshallow_noWs_noAux = trn_framework.validate_model(ShallowFullyConnectedArchitecture)
    history_FCshallow_yesWs_noAux = trn_framework.validate_model(lambda: ShallowFullyConnectedArchitecture(weight_sharing=True))
    trn_framework.enable_aux_task(True)
    history_FCshallow_noWs_yesAux = trn_framework.validate_model(ShallowFullyConnectedArchitecture)
    history_FCshallow_yesWs_yesAux = trn_framework.validate_model(lambda: ShallowFullyConnectedArchitecture(weight_sharing=True))
    
    # Validate the four variants of the deep fully connected architecture
    trn_framework.enable_aux_task(False)
    history_FCdeep_noWs_noAux = trn_framework.validate_model(DeepFullyConnectedArchitecture)
    history_FCdeep_yesWs_noAux = trn_framework.validate_model(lambda: DeepFullyConnectedArchitecture(weight_sharing=True))
    trn_framework.enable_aux_task(True)
    history_FCdeep_noWs_yesAux = trn_framework.validate_model(DeepFullyConnectedArchitecture)
    history_FCdeep_yesWs_yesAux = trn_framework.validate_model(lambda: DeepFullyConnectedArchitecture(weight_sharing=True))
    
     # Validate the four variants of the convolutional architecture
    trn_framework.enable_aux_task(False)
    history_Conv_noWs_noAux = trn_framework.validate_model(ConvolutionalArchitecture)
    history_Conv_yesWs_noAux = trn_framework.validate_model(lambda: ConvolutionalArchitecture(weight_sharing=True))
    trn_framework.enable_aux_task(True)
    history_Conv_noWs_yesAux = trn_framework.validate_model(ConvolutionalArchitecture)
    history_Conv_yesWs_yesAux = trn_framework.validate_model(lambda: ConvolutionalArchitecture(weight_sharing=True))
    
    # Compute results
    s_nn = validation_results(history_FCshallow_noWs_noAux)
    s_yn = validation_results(history_FCshallow_yesWs_noAux)
    s_ny = validation_results(history_FCshallow_noWs_yesAux)
    s_yy = validation_results(history_FCshallow_yesWs_yesAux)
    d_nn = validation_results(history_FCdeep_noWs_noAux)
    d_yn = validation_results(history_FCdeep_yesWs_noAux)
    d_ny = validation_results(history_FCdeep_noWs_yesAux)
    d_yy = validation_results(history_FCdeep_yesWs_yesAux)
    c_nn = validation_results(history_Conv_noWs_noAux)
    c_yn = validation_results(history_Conv_yesWs_noAux)
    c_ny = validation_results(history_Conv_noWs_yesAux)
    c_yy = validation_results(history_Conv_yesWs_yesAux)
    
    # Print results
    log = Logger(level='warn')
    print()
    log.warn("Miniproject 1 - Average results of the 12 models")
    print("\tShallow[n WS, n AUX]\t acc = %s +- %s \t time = %s" % (s_nn[0], s_nn[1], s_nn[2]))
    print("\tShallow[y WS, n AUX]\t acc = %s +- %s \t time = %s" % (s_yn[0], s_yn[1], s_yn[2]))
    print("\tShallow[n WS, y AUX]\t acc = %s +- %s \t time = %s" % (s_ny[0], s_ny[1], s_ny[2]))
    print("\tShallow[y WS, y AUX]\t acc = %s +- %s \t time = %s" % (s_yy[0], s_yy[1], s_yy[2]))
    print("\tDeep[n WS, n AUX]\t acc = %s +- %s \t time = %s" % (d_nn[0], d_nn[1], d_nn[2]))
    print("\tDeep[y WS, n AUX]\t acc = %s +- %s \t time = %s" % (d_yn[0], d_yn[1], d_yn[2]))
    print("\tDeep[n WS, y AUX]\t acc = %s +- %s \t time = %s" % (d_ny[0], d_ny[1], d_ny[2]))
    print("\tDeep[y WS, y AUX]\t acc = %s +- %s \t time = %s" % (d_yy[0], d_yy[1], d_yy[2]))
    print("\tConvNet[n WS, n AUX]\t acc = %s +- %s \t time = %s" % (c_nn[0], c_nn[1], c_nn[2]))
    print("\tConvNet[y WS, n AUX]\t acc = %s +- %s \t time = %s" % (c_yn[0], c_yn[1], c_yn[2]))
    print("\tConvNet[n WS, y AUX]\t acc = %s +- %s \t time = %s" % (c_ny[0], c_ny[1], c_ny[2]))
    print("\tConvNet[y WS, y AUX]\t acc = %s +- %s \t time = %s" % (c_yy[0], c_yy[1], c_yy[2]))        


if __name__ == '__main__':
    main()