import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt


LINES = ['-', ':', '--'] # line style for plots - all caps bc constant

def main():
    strings_by_author = dict()
    strings_by_author['doyle'] = text_to_string('hound.txt')
    strings_by_author['wells'] = text_to_string('war.txt')
    strings_by_author['unknown'] = text_to_string('lost.txt')

    # print(strings_by_author['doyle'][:300])
    # print(strings_by_author['wells'][:300])
    # print(strings_by_author['unknown'][:300])

    words_by_author = make_word_dict(strings_by_author)
    len_shortest_corpus = find_shortest_corpus(words_by_author)
    
    # word_length_test(words_by_author, len_shortest_corpus)
    # stopwords_test(words_by_author, len_shortest_corpus)
    # parts_of_speech_test(words_by_author, len_shortest_corpus)
    # vocab_test(words_by_author)
    # jaccard_test(words_by_author, len_shortest_corpus)

def text_to_string(filename):
    with open(filename,'r') as file:
        filename = file.read()
    return filename

def make_word_dict(strings_by_author):
    """Return dictionary of tokenized words by corpus by author"""
    words_by_author = dict()
    for author in strings_by_author:
        tokens = nltk.word_tokenize(strings_by_author[author])
        words_by_author[author] = ([token.lower() for token in tokens if token.isalpha()]) # this last bit is called list comprehension
    return words_by_author

def find_shortest_corpus(words_by_author):
    """Return length of shortest corpus"""
    word_count =[]
    for author in words_by_author:
        word_count.append(len(words_by_author[author]))
        print('\nNumber of words for {} = {}\n'.format(author, len(words_by_author[author])))
        len_shortest_corpus = min(word_count)
        print('Length of shortest corpus = {}\n'.format(len_shortest_corpus))
        return len_shortest_corpus

def word_length_test(words_by_author, len_shortest_corpus):
    """sdsds"""

def stopwords_test(words_by_author, len_shortest_corpus):
    """Ssdsd"""

def parts_of_speech_test(words_by_author, len_shortest_corpus):
    """sdfsd"""

def vocab_test(words_by_author):
    """ddd"""

def jaccard_test(words_by_author, len_shortest_corpus):
    """ddd"""



main()

