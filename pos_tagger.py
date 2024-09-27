from multiprocessing import Pool
import numpy as np
import time
from utils import *
from myconstants import *



""" Contains the part of speech tagger class. """


def evaluate(data, model):
    """Evaluates the POS model on some sentences and gold tags.

    This model can compute a few different accuracies:
        - whole-sentence accuracy
        - per-token accuracy
        - compare the probabilities computed by different styles of decoding

    You might want to refactor this into several different evaluation functions,
    or you can use it as is. 
    
    As per the write-up, you may find it faster to use multiprocessing (code included). 
    
    """
    processes = 4
    sentences = data[0]
    tags = data[1]
    n = len(sentences)
    k = n//processes
    n_tokens = sum([len(d) for d in sentences])
    unk_n_tokens = sum([1 for s in sentences for w in s if w not in model.word2idx.keys()])
    predictions = {i:None for i in range(n)}
    probabilities = {i:None for i in range(n)}
         
    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(pool.apply_async(infer_sentences, [model, sentences[i:i+k], i]))
    ans = [r.get(timeout=None) for r in res]
    predictions = dict()
    for a in ans:
        predictions.update(a)
    print(f"Inference Runtime: {(time.time()-start)/60} minutes.")
    
    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(pool.apply_async(compute_prob, [model, sentences[i:i+k], tags[i:i+k], i]))
    ans = [r.get(timeout=None) for r in res]
    probabilities = dict()
    for a in ans:
        probabilities.update(a)
    print(f"Probability Estimation Runtime: {(time.time()-start)/60} minutes.")


    token_acc = sum([1 for i in range(n) for j in range(len(sentences[i])) if tags[i][j] == predictions[i][j]]) / n_tokens
    unk_token_acc = sum([1 for i in range(n) for j in range(len(sentences[i])) if tags[i][j] == predictions[i][j] and sentences[i][j] not in model.word2idx.keys()]) / unk_n_tokens
    whole_sent_acc = 0
    num_whole_sent = 0
    for k in range(n):
        sent = sentences[k]
        eos_idxes = indices(sent, '.')
        start_idx = 1
        end_idx = eos_idxes[0]
        for i in range(1, len(eos_idxes)):
            whole_sent_acc += 1 if tags[k][start_idx:end_idx] == predictions[k][start_idx:end_idx] else 0
            num_whole_sent += 1
            start_idx = end_idx+1
            end_idx = eos_idxes[i]
    print("Whole sent acc: {}".format(whole_sent_acc/num_whole_sent))
    print("Mean Probabilities: {}".format(sum(probabilities.values())/n))
    print("Token acc: {}".format(token_acc))
    print("Unk token acc: {}".format(unk_token_acc))
    
    confusion_matrix(pos_tagger.tag2idx, pos_tagger.idx2tag, predictions.values(), tags, 'cm.png')

    return whole_sent_acc/num_whole_sent, token_acc, sum(probabilities.values())/n


class POSTagger():
    def __init__(self):
        """Initializes the tagger model parameters and anything else necessary. """
        pass

    def add_k_smoothing(self, row, k):
        rowsum = sum(row)
        row = (row + k) / (rowsum + k*len(row))
        return row


    def add_1_smoothing(self, row):
         return self.add_k_smoothing(row, 1)
    
    
    def get_unigrams(self):
        """
        Computes unigrams. 
        Tip. Map each tag to an integer and store the unigrams in a numpy array. 
        """
        tag_sentences = self.data[1]
        num_tags = len(self.all_tags)
        unigrams = np.zeros(num_tags)
        for tag_sent in tag_sentences:
            for tag in tag_sent:
                unigrams[self.tag2idx[tag]] += 1
        
        unigrams = self.add_k_smoothing(unigrams, SMOOTHING_K)
        return unigrams


    def get_bigrams(self):        
        """
        Computes bigrams. 
        Tip. Map each tag to an integer and store the bigrams in a numpy array
             such that bigrams[index[tag1], index[tag2]] = Prob(tag2|tag1). 
        """
        tag_sentences = self.data[1]
        num_tags = len(self.all_tags)
        bigrams = np.zeros([num_tags, num_tags])
        for tag_sent in tag_sentences:
            sentence_length = len(tag_sent)
            for i in range(sentence_length - 1):
                tag1 = tag_sent[i]
                tag2 = tag_sent[i+1]
                bigrams[self.tag2idx[tag1]][self.tag2idx[tag2]] += 1
        print(bigrams)

        for i in range(num_tags):
            bigrams[i] = self.add_k_smoothing(bigrams[i], SMOOTHING_K)

        print(bigrams)

        return bigrams

    
    def get_trigrams(self):
        """
        Computes trigrams. 
        Tip. Similar logic to unigrams and bigrams. Store in numpy array. 
        """
        tag_sentences = self.data[1]
        num_tags = len(self.all_tags)
        trigrams = np.zeros([num_tags, num_tags, num_tags])
        for tag_sent in tag_sentences:
            sentence_length = len(tag_sent)
            for i in range(sentence_length - 2):
                tag1 = tag_sent[i]
                tag2 = tag_sent[i+1]
                tag3 = tag_sent[i+2]
                trigrams[self.tag2idx[tag1]][self.tag2idx[tag2]][self.tag2idx[tag3]] += 1

        for i in range(num_tags):
            for j in range(num_tags):
                trigrams[i][j] = self.add_k_smoothing(trigrams[i][j], SMOOTHING_K)

        return trigrams
    
    
    def get_emissions(self):
        """
        Computes emission probabilities. 
        Tip. Map each tag to an integer and each word in the vocabulary to an integer. 
             Then create a numpy array such that lexical[index(tag), index(word)] = Prob(word|tag) 
        """
        words, tags = self.data
        num_docs = len(words)
        len_vocab = len(self.vocabulary)
        num_tags = len(self.all_tags)
        emissions = np.zeros([num_tags, len_vocab])

        for i in range(num_docs):
            sentence_words = words[i]
            sentence_tags = tags[i]
            for j in range(len(sentence_words)):
                word = sentence_words[j]
                tag = sentence_tags[j]
                emissions[self.tag2idx[tag]][self.word2idx[word]] += 1

        emissions = emissions / emissions.sum(axis=1)[:, None]
        return emissions

    

    def train(self, data):
        """Trains the model by computing transition and emission probabilities.

        You should also experiment:
            - smoothing.
            - N-gram models with varying N.
        
        """
        self.data = data
        self.all_tags = list(set([t for tag in data[1] for t in tag]))
        self.tag2idx = {self.all_tags[i]:i for i in range(len(self.all_tags))}
        self.idx2tag = {v:k for k,v in self.tag2idx.items()}
        self.vocabulary = list(set(word for sentence in data[0] for word in sentence))
        self.word2idx = {self.vocabulary[i]: i for i in range(len(self.vocabulary))}
        self.transition = self.get_bigrams()
        self.emission = self.get_emissions()


    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition
        probabilities.
        """
        ## TODO
        return 0.

    def inference(self, sequence):
        """Tags a sequence with part of speech tags.

        You should implement different kinds of inference (suggested as separate
        methods):

            - greedy decoding
            - decoding with beam search
            - viterbi
        """
        ## TODO
        return []


if __name__ == "__main__":
    pos_tagger = POSTagger()

    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")

    pos_tagger.train(train_data)

    # # Experiment with your decoder using greedy decoding, beam search, viterbi...

    # # Here you can also implement experiments that compare different styles of decoding,
    # # smoothing, n-grams, etc.
    # evaluate(dev_data, pos_tagger)

    # # Predict tags for the test set
    # test_predictions = []
    # for sentence in test_data:
    #     test_predictions.extend(pos_tagger.inference(sentence))
    
    # # Write them to a file to update the leaderboard
    # # TODO
