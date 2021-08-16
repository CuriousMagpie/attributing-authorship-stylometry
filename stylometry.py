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
    
    word_length_test(words_by_author, len_shortest_corpus)
    stopwords_test(words_by_author, len_shortest_corpus)
    parts_of_speech_test(words_by_author, len_shortest_corpus)
    vocab_test(words_by_author)
    jaccard_test(words_by_author, len_shortest_corpus)

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
    """Plit word length frequency by author, truncated to shortest corpus length"""
    by_author_length_freq_dist = dict()
    plt.figure(1)
    plt.ion()

    for i, author in enumerate(words_by_author):
        word_lengths = [len(word) for word in words_by_author[author] [:len_shortest_corpus]] #list comprehension again
        by_author_length_freq_dist[author] = nltk.FreqDist(word_lengths)
        by_author_length_freq_dist[author].plot(10, linestyle=LINES[i], label=author, title='Word Length')
        # can use cumulative=True to generate a cumulative distribution
    
    plt.legend()
    plt.savefig('figure-1.png')
    # plt.show(block=True)

def stopwords_test(words_by_author, len_shortest_corpus):
    """Plot stopwords frequency by author, truncated to shortest corpus length"""
    stopwords_by_author_freq_dist = dict()
    plt.figure(2)
    stop_words = set(stopwords.words('english')) #use set for speed
    print('Number of stopwords = {}\n'.format(len(stop_words)))
    # print('Stopwords = {}\n'.format(stop_words))

    for i, author in enumerate(words_by_author):
        stopwords_by_author = [word for word in words_by_author[author] [:len_shortest_corpus] if word in stop_words]
        stopwords_by_author_freq_dist[author] = nltk.FreqDist(stopwords_by_author)
        stopwords_by_author_freq_dist[author].plot(25, label=author, linestyle=LINES[i], title='25 Most Common Stopwords')

    plt.legend()
    plt.savefig('figure-2.png')
    # plt.show(block=True)

def parts_of_speech_test(words_by_author, len_shortest_corpus):
    """Plot author use of parts of speech, ie, nouns, verbs, pronouns"""
    by_author_pos_dist = dict()
    plt.figure(3)

    for i, author, in enumerate(words_by_author):
        pos_by_author = [pos[1] for pos in nltk.pos_tag(words_by_author[author] [:len_shortest_corpus])]
        by_author_pos_dist[author] = nltk.FreqDist(pos_by_author)
        by_author_pos_dist[author].plot(20, label=author, linestyle=LINES[i], title='25 Most Common Parts of Speech')


    plt.legend()
    plt.savefig('figure-3.png')
    # plt.show(block=True)

def vocab_test(words_by_author):
    """Compare author vocabularies by using the chi-squared statistical test"""
    chi_squared_by_author = dict()
    for author in words_by_author:
        if author != 'unknown':
            combined_corpus = (words_by_author[author] + words_by_author['unknown'])
            author_proportion = (len(words_by_author[author]) / len(combined_corpus))
            combined_freq_dist = nltk.FreqDist(combined_corpus)
            most_common_words = list(combined_freq_dist.most_common(1000))
            chisquared = 0
            for word, combined_count in most_common_words:
                observed_count_author = words_by_author[author].count(word)
                expected_count_author = combined_count * author_proportion
                chisquared += ((observed_count_author - expected_count_author)**2 / expected_count_author)
                chi_squared_by_author[author] = chisquared
            print('Chi-squared for {} = {:.1f}'.format(author, chisquared))
    most_likely_author = min(chi_squared_by_author, key=chi_squared_by_author.get)
    print('Most likely author by vocabulary is {}\n'.format(most_likely_author))


def jaccard_test(words_by_author, len_shortest_corpus):
    """Calculate Jaccard similarity of each known corpus to unknown corpus"""
    jaccard_by_author = dict()
    unique_words_unknown = set(words_by_author['unknown'] [:len_shortest_corpus])
    authors = (author for author in words_by_author if author != 'unknown') #generator expression
    for author in authors:
        unique_words_author = set(words_by_author[author] [:len_shortest_corpus])
        shared_words = unique_words_author.intersection(unique_words_unknown)
        jaccard_sim = (float(len(shared_words)) / (len(unique_words_author) + len(unique_words_unknown)- len(shared_words)))
        jaccard_by_author[author] = jaccard_sim
        print('Jaccard Similarity for {} = {}'.format(author, jaccard_sim))
    most_likely_author = max(jaccard_by_author, key=jaccard_by_author.get)
    print('Most likely author by similarity is {}'.format(most_likely_author))


if __name__ == '__main__':
    main()
