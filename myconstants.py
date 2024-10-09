### Append stop word ###
STOP_WORD = True
### Capitalization
CAPITALIZATION = True

### small number
EPSILON = 1e-100

## smoothing constants
SMOOTHING_K = 1e-7
EMISSION_K = 1e-10

### Inference Types ###
GREEDY = 0
BEAM = 1; BEAM_K = 3
VITERBI = 2
INFERENCE = VITERBI

### Smoothing Types ###
LAPLACE = 0; LAPLACE_FACTOR = .2
INTERPOLATION = 1; LAMBDAS =  [0.05,0.15, 0.8]
ADD_K = 2 
SMOOTHING = INTERPOLATION


# NGRAMM
NGRAMM = 3

## Handle unknown words TnT style
TNT_UNK = True
UNK_C = 10 #words with count to be considered
UNK_M = 10 #substring length to be considered


# Set to True to run the model on the test set
RUN_TEST = True