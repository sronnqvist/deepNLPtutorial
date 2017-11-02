from nltk.tokenize import word_tokenize
import csv


def preproc(text):
    """ Text preprocessing and tokenization """
    tokens = word_tokenize(text.replace("<br />", "\n"))
    return tokens


def read_file(file_name, limit=None):
    """ Read CSV data file """
    print("Reading", file_name)
    data = []
    csvfile = open(file_name, 'r')
    for i, line in enumerate(csv.DictReader(csvfile, delimiter="\t")):
        if limit and i > limit:
            break
        if i % 1000 == 999:
            print(i+1, "comments")
        line['tokens'] = preproc(line['review'])
        if 'sentiment' in line:
            line['sentiment'] = int(line['sentiment'])
        data.append(line)
    return data
