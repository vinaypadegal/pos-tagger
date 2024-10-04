from multiprocessing import Pool
import numpy as np
import time
from utils import *

import sys
import csv
import math
from myconstants import *


""" Contains the part of speech tagger class. """
class SuffixTree:
    def __init__(self):
        self.suffixes = {}

    def add_word(self, word, tag):
        """Add word and its tag to the suffix tree."""
        for i in range(len(word)):
            suffix = word[i:]
            if suffix not in self.suffixes:
                self.suffixes[suffix] = {tag: 1}
            else:
                if tag in self.suffixes[suffix]:
                    self.suffixes[suffix][tag] += 1
                else:
                    self.suffixes[suffix][tag] = 1

    def get_suffix_probabilities(self, word):
        """Get probabilities of tags based on the word's suffix."""
        for i in range(len(word)):
            suffix = word[i:]
            if suffix in self.suffixes:
                tag_counts = self.suffixes[suffix]
                total_counts = sum(tag_counts.values())
                return {tag: count / total_counts for tag, count in tag_counts.items()}
        return None

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


    if len(predictions) == len(tags):
        flag = 0
        for i in range(len(tags)):
            if len(predictions[i]) != len(tags[i]):
                print("Error: Predictions and tags have different lengths for i - ", i)
                flag = 1 
        if flag == 1:
            return
    else:
        print("Error: Predictions and tags have different lengths.")
        return

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
        self.suffix_tree = SuffixTree()
        

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
        for i in range(num_tags):
            bigrams[i] = self.add_k_smoothing(bigrams[i], SMOOTHING_K)
            
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

        # emissions = emissions / emissions.sum(axis=1)[:, None]
        for i in range(num_tags):
            emissions[i] = self.add_k_smoothing(emissions[i], EMISSION_K)

        return emissions


    def get_suffix_emission(self, word):
        # Get emission probabilities for an unknown word using suffix tree.
        # This function returns an array of emission probabilities for all tags
        suffix_probs = self.suffix_tree.get_suffix_probabilities(word)
        if suffix_probs:
            # Convert the suffix-based tag probabilities to emission probabilities
            emission = np.zeros(len(self.all_tags))
            for tag, prob in suffix_probs.items():
                emission[self.tag2idx[tag]] = prob

            emission = self.add_k_smoothing(emission, EMISSION_K)
            return emission
        else:
            # If no suffix match, return uniform probability, Note: something better can be done?
            return np.ones(len(self.all_tags)) / len(self.all_tags)


    def get_emission_prob(self, word, tag):
        # Get emission probability for a word-tag pair
        if word in self.word2idx:
            # Known word: Use standard emission probabilities
            emission_prob = self.emission[self.tag2idx[tag], self.word2idx[word]]
        else:
            # Unknown word: Use suffix-based emission probabilities
            suffix_emission = self.get_suffix_emission(word)
            emission_prob = suffix_emission[self.tag2idx[tag]]
        
        return emission_prob
    

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

        # Build the suffix tree
        for i in range(len(data[0])):
            for word, tag in zip(data[0][i], data[1][i]):
                self.suffix_tree.add_word(word, tag)
        
        self.bigrams = self.get_bigrams()
        self.trigrams = self.get_trigrams()
        self.transition = self.trigrams if NGRAMM == 3 else self.bigrams
        self.emission = self.get_emissions()



    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition
        probabilities.
        """
        n = len(sequence)
        score = 1

        if (self.transition == self.bigrams).all():
            for i in range(1,n):
                score *= self.transition[self.tag2idx[tags[i-1]],self.tag2idx[tags[i]]] * self.get_emission_prob(sequence[i], tags[i])

        elif (self.transition == self.trigrams).all():
            for i in range(2,n):
                score *= self.transition[self.tag2idx[tags[i-2]],self.tag2idx[tags[i-1]],self.tag2idx[tags[i]]] * self.get_emission_prob(sequence[i], tags[i])
        else:
            print("Transition matrix not defined.")
            return None

        
        return score
    

    def beam_search(self, sequence, k=20):
        
        n = len(sequence)
        
        if (self.transition == self.bigrams).all():

            # print("Beam Search with Bigrams")

            # Initialize the beam with the  possible tags for the first word, possible scores
            beam = [([tag], self.emission[self.tag2idx[tag],self.word2idx.get(sequence[0], -1)]) for tag in self.all_tags]
            beam = sorted(beam, key=lambda x: x[1], reverse=True)[:k]
            

            # For each subsequent word in the sequence, note : n-1 to avoid <STOP> tag
            for t in range(1, n-1):
                new_beam = []
                
                # For each tag sequence in the current beam
                for tags, score in beam:
                    last_tag = tags[-1]

                

                    next_tag_probs = [
                        (next_tag, self.transition[self.tag2idx[last_tag],self.tag2idx[next_tag]] * self.get_emission_prob(sequence[t], next_tag))
                        for next_tag in self.all_tags
                    ]
                    
                    # Choose the top k next tags based on transition * emission probabilities
                    best_next_tags = sorted(next_tag_probs, key=lambda x: x[1], reverse=True)[:k]
                    
                    # Calculate scores for the top k next tags by multiplying with the current score 
                    for next_tag, total_prob in best_next_tags:
                        
                        # Create new tag sequence
                        new_tags = tags + [next_tag]
                        new_beam.append((new_tags, score * total_prob))

                # At every step keep only the k best tag sequences
                beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:k]

            # Return the tag sequence with the highest probability
            highest_prob = beam[0][0]
            highest_prob.append('<STOP>') # Appending stop as the last tag of the sentence
            return highest_prob
        
        elif (self.transition == self.trigrams).all():

            # print("Beam Search with Trigrams")

            beam = [([tag1, tag2], self.get_emission_prob(sequence[0], tag1) * self.get_emission_prob(sequence[1], tag2))
                for tag1 in self.all_tags for tag2 in self.all_tags]
            beam = sorted(beam, key=lambda x: x[1], reverse=True)[:k]

            # For each subsequent word in the sequence, apply trigram transitions
            for t in range(2, n-1):
                new_beam = []

                for tags, score in beam:
                    last_tag1 = tags[-2]
                    last_tag2 = tags[-1]

                    next_tag_probs = [
                        (next_tag, self.transition[self.tag2idx[last_tag1], self.tag2idx[last_tag2], self.tag2idx[next_tag]] * self.get_emission_prob(sequence[t], next_tag))
                        for next_tag in self.all_tags
                    ]

                    best_next_tags = sorted(next_tag_probs, key=lambda x: x[1], reverse=True)[:k]

                    for next_tag, total_prob in best_next_tags:
                        new_tags = tags + [next_tag]
                        new_beam.append((new_tags, score * total_prob))

                beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:k]

            highest_prob = beam[0][0]
            highest_prob.append('<STOP>')
            return highest_prob
        
        else:
            print("Transition matrix not defined.")
            return None
        

    def viterbi(self, sentence):
        if NGRAMM == 2:
            return self.bigram_viterbi(sentence)
        elif NGRAMM == 3:
            return self.trigram_viterbi(sentence)
        else:
            print("We'll see about the extra credit")


    def bigram_viterbi(self, sentence):
        n = len(sentence)
        num_tags = len(self.all_tags)
        lattice = np.zeros([num_tags, n])
        backpointers = np.zeros([num_tags, n], dtype=int)

        start_idx = self.tag2idx['O']

        # all tag sequences must start with <START>
        for j in range(num_tags):
            lattice[j][1] = math.log(self.transition[start_idx][j]) + math.log(self.get_emission_prob(sentence[1], self.idx2tag[j]))
            backpointers[j][1] = start_idx

        for k in range(2, n):
            for j in range(num_tags):
                max_prob = -float('inf')
                bp = -1
                for i in range(num_tags):
                    prob = lattice[i, k-1] + math.log(self.transition[i, j]) + math.log(self.get_emission_prob(sentence[k], self.idx2tag[j]))
                    if prob > max_prob:
                        max_prob = prob
                        bp = i
                lattice[j][k] = max_prob
                backpointers[j][k] = bp

        # finding best tag sequence
        tag_seq = []
        backindex = self.tag2idx['<STOP>']
        k = n-1
        while k >= 0:
            tag_seq.append(self.idx2tag[backindex])
            backindex = backpointers[backindex][k]
            k -= 1

        return tag_seq[::-1]


    def trigram_viterbi(self, sentence):
        n = len(sentence)
        num_tags = len(self.all_tags)
        lattice = np.zeros([num_tags * num_tags, n])
        backpointers = np.zeros([num_tags * num_tags, n], dtype=int)

        start_idx = self.tag2idx['O']
        start = (start_idx * num_tags) + start_idx

        # all tag sequences must start with <START><START>
        start_t_idx = num_tags * self.tag2idx['O']  # index where tag combination is <start>, <t>
        for j in range(0, num_tags - 1):
            lattice[start_t_idx + j][1] = math.log(self.transition[start_idx, start_idx, j]) + math.log(self.get_emission_prob(sentence[1], self.idx2tag[j]))
            backpointers[start_t_idx + j][1] = start

        for k in range(2, n):
            for j in range(num_tags * num_tags):
                prev = j // num_tags
                cur = j % num_tags
                max_prob = -float('inf')
                bp = -1
                for i in range(prev, num_tags * num_tags, num_tags):
                    if lattice[i, k-1] == 0:
                        continue
                    prevprev = i // num_tags
                    prob = lattice[i, k-1] + math.log(self.transition[prevprev, prevprev, cur]) + math.log(self.get_emission_prob(sentence[k], self.idx2tag[cur]))
                    if prob > max_prob:
                        max_prob = prob
                        bp = i
                lattice[j][k] = max_prob
                backpointers[j][k] = bp

        # finding best tag sequence
        tag_seq = []
        stop_idx = self.tag2idx['<STOP>']
        backindex = np.argmax(lattice[:, n-1][stop_idx:num_tags*num_tags:num_tags])
        backindex = stop_idx + (backindex * num_tags)
        k = n-1
        while k >= 0:
            cur = backindex % num_tags
            tag_seq.append(self.idx2tag[cur])
            backindex = backpointers[backindex][k]
            k -= 1

        return tag_seq[::-1]

        
        

    def inference(self, sequence):
        """Tags a sequence with part of speech tags.

        You should implement different kinds of inference (suggested as separate
        methods):

            - greedy decoding
            - decoding with beam search
            - viterbi
        """
        # Call k beams below
        # k = BEAM_K
        # beam_search_seq = self.beam_search(sequence, k)
        # return beam_search_seq
    
        # Call viterbi below
        viterbi = self.viterbi(sequence)
        return viterbi
        
        
      


if __name__ == "__main__":
    pos_tagger = POSTagger()

    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")
    dev2_data = load_data("data/dev2_x.csv", "data/dev2_y.csv")
    pos_tagger.train(train_data)

    # # Experiment with your decoder using greedy decoding, beam search, viterbi...

    # # Here you can also implement experiments that compare different styles of decoding,
    # # smoothing, n-grams, etc.
    
    evaluate(dev_data, pos_tagger)
    

    # Predict tags for the test set
    test_predictions = []
    for sentence in tqdm(test_data):
        
        test_predictions.extend(pos_tagger.inference(sentence)[:-1])
        
    # # print(len(test_predictions))
    
    # # # # Write them to a file to update the leaderboard
    test_predictions = pd.DataFrame(test_predictions)
    test_predictions.to_csv("test_y.csv", index=True,index_label=['id'], header=['tag'],quoting=csv.QUOTE_NONNUMERIC)
    
