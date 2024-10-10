To set method of inference, in constants.py set 

INFERENCE = BEAM or INFERENCE = VITERBI or INFERENCE = GREEDY 

To set the N-gram model, in constants.py set

NGRAMM = 2 or NGRAMM = 3 

To set method of smoothing, in constants.py set: 

SMOOTHING = ADD_K or SMOOTHING = INTERPOLATION 

For ADD_K smoothing, SMOOTHING_K is the smoothing constant for transitions and EMISSION_K is the smoothing contant for emissions. These are by default set to the best values obtained after experimentation. 

For add_1 smoothing set:

SMOOTHING = ADD_K and SMOOTHING_K = 1


To run the model on the test set, in constants.py set:

RUN_TEST = True

The best results were achieved using INFERENCE = VITERBI, SMOOTHING = INTERPOLATION, LAMBDAS = [0.05,0.15,0.80], NGRAM = 3

To run the code:

First ensure that the data is stored in ./data/. 

Set the constants as you wish 

Run: python3 pos_tagger.py
