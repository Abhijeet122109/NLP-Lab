'''Name:Abhijeet Landage
Batch:B2
Roll no:36
Pract no 3: Generating the n gram model using nltk'''
from nltk import ngrams

from nltk.util import ngrams
#unigram model
n = 1
sentence = 'While unigram model sentences will only exclude the UNK token, models will also exclude all other words already in the sentence.NTK provides another function everygrams that converts a sentence into unigram, bigram, trigram, and so on till the ngrams, where n is the length of the sentence. In short, this function generates ngrams for all possible values of n.'
unigrams = ngrams(sentence.split(), n)

for item in unigrams:
    print(item)
#bigram model
n = 2
sentence = 'While unigram model sentences will only exclude the UNK token, models will also exclude all other words already in the sentence.NTK provides another function everygrams that converts a sentence into unigram, bigram, trigram, and so on till the ngrams, where n is the length of the sentence. In short, this function generates ngrams for all possible values of n.'
unigrams = ngrams(sentence.split(), n)

for item in unigrams:
    print(item)
#trigram model
n = 3
sentence = 'While unigram model sentences will only exclude the UNK token, models will also exclude all other words already in the sentence.NTK provides another function everygrams that converts a sentence into unigram, bigram, trigram, and so on till the ngrams, where n is the length of the sentence. In short, this function generates ngrams for all possible values of n.'
unigrams = ngrams(sentence.split(), n)

for item in unigrams:
    print(item)

#using text file input
from nltk import ngrams
file = open("/home/exam/Desktop/NLP_LAB75/al.txt")
for i in file.readlines():
    cumulative = i
    sentences = i.split(".")
    counter = 0
    for sentence in sentences:
        print("For sentence", counter + 1, ", trigrams are: ")
        trigrams = ngrams(sentence.split(" "), 3)
        for grams in trigrams:
            print(grams)
        counter += 1
        print()
        
 #output    
'''('While',)
('unigram',)
('model',)
('sentences',)
('will',)
('only',)
('exclude',)
('the',)
('UNK',)
('token,',)
('models',)
('will',)
('also',)
('exclude',)
('all',)
('other',)
('words',)
('already',)
('in',)
('the',)
('sentence.NTK',)
('provides',)
('another',)
('function',)
('everygrams',)
('that',)
('converts',)
('a',)
('sentence',)
('into',)
('unigram,',)
('bigram,',)
('trigram,',)
('and',)
('so',)
('on',)
('till',)
('the',)
('ngrams,',)
('where',)
('n',)
('is',)
('the',)
('length',)
('of',)
('the',)
('sentence.',)
('In',)
('short,',)
('this',)
('function',)
('generates',)
('ngrams',)
('for',)
('all',)
('possible',)
('values',)
('of',)
('n.',)

#bigram model
('While', 'unigram')
('unigram', 'model')
('model', 'sentences')
('sentences', 'will')
('will', 'only')
('only', 'exclude')
('exclude', 'the')
('the', 'UNK')
('UNK', 'token,')
('token,', 'models')
('models', 'will')
('will', 'also')
('also', 'exclude')
('exclude', 'all')
('all', 'other')
('other', 'words')
('words', 'already')
('already', 'in')
('in', 'the')
('the', 'sentence.NTK')
('sentence.NTK', 'provides')
('provides', 'another')
('another', 'function')
('function', 'everygrams')
('everygrams', 'that')
('that', 'converts')
('converts', 'a')
('a', 'sentence')
('sentence', 'into')
('into', 'unigram,')
('unigram,', 'bigram,')
('bigram,', 'trigram,')
('trigram,', 'and')
('and', 'so')
('so', 'on')
('on', 'till')
('till', 'the')
('the', 'ngrams,')
('ngrams,', 'where')
('where', 'n')
('n', 'is')
('is', 'the')
('the', 'length')
('length', 'of')
('of', 'the')
('the', 'sentence.')
('sentence.', 'In')
('In', 'short,')
('short,', 'this')
('this', 'function')
('function', 'generates')
('generates', 'ngrams')
('ngrams', 'for')
('for', 'all')
('all', 'possible')
('possible', 'values')
('values', 'of')
('of', 'n.')
#tri-gram model
('While', 'unigram', 'model')
('unigram', 'model', 'sentences')
('model', 'sentences', 'will')
('sentences', 'will', 'only')
('will', 'only', 'exclude')
('only', 'exclude', 'the')
('exclude', 'the', 'UNK')
('the', 'UNK', 'token,')
('UNK', 'token,', 'models')
('token,', 'models', 'will')
('models', 'will', 'also')
('will', 'also', 'exclude')
('also', 'exclude', 'all')
('exclude', 'all', 'other')
('all', 'other', 'words')
('other', 'words', 'already')
('words', 'already', 'in')
('already', 'in', 'the')
('in', 'the', 'sentence.NTK')
('the', 'sentence.NTK', 'provides')
('sentence.NTK', 'provides', 'another')
('provides', 'another', 'function')
('another', 'function', 'everygrams')
('function', 'everygrams', 'that')
('everygrams', 'that', 'converts')
('that', 'converts', 'a')
('converts', 'a', 'sentence')
('a', 'sentence', 'into')
('sentence', 'into', 'unigram,')
('into', 'unigram,', 'bigram,')
('unigram,', 'bigram,', 'trigram,')
('bigram,', 'trigram,', 'and')
('trigram,', 'and', 'so')
('and', 'so', 'on')
('so', 'on', 'till')
('on', 'till', 'the')
('till', 'the', 'ngrams,')
('the', 'ngrams,', 'where')
('ngrams,', 'where', 'n')
('where', 'n', 'is')
('n', 'is', 'the')
('is', 'the', 'length')
('the', 'length', 'of')
('length', 'of', 'the')
('of', 'the', 'sentence.')
('the', 'sentence.', 'In')
('sentence.', 'In', 'short,')
('In', 'short,', 'this')
('short,', 'this', 'function')
('this', 'function', 'generates')
('function', 'generates', 'ngrams')
('generates', 'ngrams', 'for')
('ngrams', 'for', 'all')
('for', 'all', 'possible')
('all', 'possible', 'values')
('possible', 'values', 'of')
('values', 'of', 'n.')

#text file input
For sentence 1 , trigrams are: 
('Embedding', 'is', 'a')
('is', 'a', 'language')
('a', 'language', 'modeling')
('language', 'modeling', 'technique')
('modeling', 'technique', 'used')
('technique', 'used', 'for')
('used', 'for', 'mapping')
('for', 'mapping', 'words')
('mapping', 'words', 'to')
('words', 'to', 'vectors')
('to', 'vectors', 'of')
('vectors', 'of', 'real')
('of', 'real', 'numbers')

For sentence 2 , trigrams are: 
('', 'It', 'represents')
('It', 'represents', 'words')
('represents', 'words', 'or')
('words', 'or', 'phrases')
('or', 'phrases', 'in')
('phrases', 'in', 'vector')
('in', 'vector', 'space')
('vector', 'space', 'with')
('space', 'with', 'several')
('with', 'several', 'dimensions')

For sentence 3 , trigrams are: 
('', 'Word', 'embeddings')
('Word', 'embeddings', 'can')
('embeddings', 'can', 'be')
('can', 'be', 'generated')
('be', 'generated', 'using')
('generated', 'using', 'various')
('using', 'various', 'methods')
('various', 'methods', 'like')
('methods', 'like', 'neural')
('like', 'neural', 'networks,')
('neural', 'networks,', 'co-occurrence')
('networks,', 'co-occurrence', 'matrix,')
('co-occurrence', 'matrix,', 'probabilistic')
('matrix,', 'probabilistic', 'models,')
('probabilistic', 'models,', 'etc')

For sentence 4 , trigrams are: 
('', 'Word2Vec', 'consists')
('Word2Vec', 'consists', 'of')
('consists', 'of', 'models')
('of', 'models', 'for')
('models', 'for', 'generating')
('for', 'generating', 'word')
('generating', 'word', 'embedding')

For sentence 5 , trigrams are: 
('', 'These', 'models')
('These', 'models', 'are')
('models', 'are', 'shallow')
('are', 'shallow', 'two-layer')
('shallow', 'two-layer', 'neural')
('two-layer', 'neural', 'networks')
('neural', 'networks', 'having')
('networks', 'having', 'one')
('having', 'one', 'input')
('one', 'input', 'layer,')
('input', 'layer,', 'one')
('layer,', 'one', 'hidden')
('one', 'hidden', 'layer,')
('hidden', 'layer,', 'and')
('layer,', 'and', 'one')
('and', 'one', 'output')
('one', 'output', 'layer')

For sentence 6 , trigrams are: 
('', 'Word2Vec', 'utilizes')
('Word2Vec', 'utilizes', 'two')
('utilizes', 'two', 'architectures')
('two', 'architectures', ':(Continuous')
('architectures', ':(Continuous', 'Bag')
(':(Continuous', 'Bag', 'of')
('Bag', 'of', 'Words):')
('of', 'Words):', 'CBOW')
('Words):', 'CBOW', 'model')
('CBOW', 'model', 'predicts')
('model', 'predicts', 'the')
('predicts', 'the', 'current')
('the', 'current', 'word')
('current', 'word', 'given')
('word', 'given', 'context')
('given', 'context', 'words')
('context', 'words', 'within')
('words', 'within', 'a')
('within', 'a', 'specific')
('a', 'specific', 'window')

For sentence 7 , trigrams are: 
('', 'The', 'input')
('The', 'input', 'layer')
('input', 'layer', 'contains')
('layer', 'contains', 'the')
('contains', 'the', 'context')
('the', 'context', 'words')
('context', 'words', 'and')
('words', 'and', 'the')
('and', 'the', 'output')
('the', 'output', 'layer')
('output', 'layer', 'contains')
('layer', 'contains', 'the')
('contains', 'the', 'current')
('the', 'current', 'word')

For sentence 8 , trigrams are: 
('', 'The', 'hidden')
('The', 'hidden', 'layer')
('hidden', 'layer', 'contains')
('layer', 'contains', 'thenumber')
('contains', 'thenumber', 'of')
('thenumber', 'of', 'dimensions')
('of', 'dimensions', 'in')
('dimensions', 'in', 'which')
('in', 'which', 'we')
('which', 'we', 'want')
('we', 'want', 'to')
('want', 'to', 'represent')
('to', 'represent', 'the')
('represent', 'the', 'current')
('the', 'current', 'word')
('current', 'word', 'present')
('word', 'present', 'at')
('present', 'at', 'the')
('at', 'the', 'output')
('the', 'output', 'layer')

For sentence 9 , trigrams are: 
('', 'Skip', 'Gram')
('Skip', 'Gram', ':')
('Gram', ':', 'Skip')
(':', 'Skip', 'gram')
('Skip', 'gram', 'predicts')
('gram', 'predicts', 'the')
('predicts', 'the', 'surrounding')
('the', 'surrounding', 'context')
('surrounding', 'context', 'words\n')

'''
