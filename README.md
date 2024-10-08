To set method of inference, in constants.py set 

INFERENCE = BEAM or INFERENCE = VITERBI or INFERENCE = GREEDY 

To set the N-gram model, in constants.py set

NGRAMM = 2 or NGRAMM = 3 

To set method of smoothing, in constants.py set: 

SMOOTHING = ADD_K or SMOOTHING = INTERPOLATION 

For add_1 smoothing set:

SMOOTHING = ADD_K and SMOOTHING_K = 1

Default setting ie setting under which best results were obtained:

SMOOTHING = ADD_K and NGRAMM = 2 and INFERENCE = VITERBI 

To run the code:

First ensure that the data is stored in ./data/. 

Set the constants as you wish 

Run: python3 pos_tagger.py
