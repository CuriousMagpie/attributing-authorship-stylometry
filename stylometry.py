import nltk
from nltk.corpus import stopwords
import matplotlib as plt

LINES = ['-', ':', '--'] # line style for plots - all caps bc constant

def main():
    strings_by_author = dict()
