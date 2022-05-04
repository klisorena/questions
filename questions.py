import nltk
import sys
import os, string, math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    data = {}
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        # print(path)
        with open(path, encoding="utf8") as f:
            content = f.read()
        data[file] = content
    return data


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = nltk.word_tokenize(document)
    words = [word.lower() for word in words]
    stop_words = nltk.corpus.stopwords.words("english")
    words = [word for word in words if (word not in string.punctuation and word not in stop_words)]

    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    def cnt_doc(w):
        cnt = 0
        for doc in documents:
            if w in documents[doc]:
                cnt += 1
        return cnt

    data = {}
    for doc in documents:
        for word in documents[doc]:
            if word in data:
                continue
            data[word] = math.log(len(documents) / cnt_doc(word))
    return data


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    f_scores = {}
    for word in query:
        for file in files:
            if word not in files[file]:
                continue
            if file in f_scores:
                f_scores[file] += files[file].count(word) * idfs[word]
            else:
                f_scores[file] = files[file].count(word) * idfs[word]

    sort_scores = dict(sorted(f_scores.items(), key=lambda x: x[1], reverse=True))
    f_list = list(sort_scores.keys())
    return f_list[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    s_scores = {}

    for word in query:
        for sentence in sentences:
            if word not in sentences[sentence]:
                continue
            if sentence in s_scores:
                s_scores[sentence] += idfs[word]
            else:
                s_scores[sentence] = idfs[word]

    sort_scores = dict(sorted(s_scores.items(), key=lambda x: x[1], reverse=True))
    f_list = list(sort_scores.keys())

    # check for duplicate
    rev_dict = {}
    for key, value in s_scores.items():
        rev_dict.setdefault(value, set()).add(key)

    duplicate = list(filter(lambda x: len(x) > 1, rev_dict.values()))

    if len(duplicate) != 0:
        qtd_scores = {}
        for sentence in sentences:
            term_cnt = 0
            for term in query:
                if term not in sentences[sentence]:
                    continue
                term_cnt += 1
            qtd_scores[sentence] = term_cnt / len(sentences[sentence])
        n_list = []
        for sentence in f_list:
            if sentence in n_list:
                continue
            if sentence not in duplicate:
                n_list.append(sentence)
            else:
                # get a list of duplicated values from sort_scores
                dup_vals = []
                for item in duplicate:
                    dup_vals.append(sort_scores[item])
                vals = set(dup_vals)
                # for each value, get a list of all keys & # sort keys according to qtd
                for val in vals:
                    temp_tie = {}
                    for s in sort_scores:
                        if val == sort_scores[s]:
                            temp_tie[s] = qtd_scores[s]
                    sort_ties = dict(sorted(temp_tie.items(), key=lambda x: x[1], reverse=True))
                    ties = list(sort_ties.keys())
                    for tie in ties:
                        if tie in n_list:
                            continue
                        n_list.append(tie)
                return n_list[:n]

    return f_list[:n]


if __name__ == "__main__":
    main()
