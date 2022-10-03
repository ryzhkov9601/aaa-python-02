from collections import Counter


class CountVectorizer:
    """
    A class to represent a count vectorizer's functionality.
    """

    def __init__(self):
        self._feature_names = []

    def fit_transform(self, corpus):
        """
        Fits for given corpus, then return count matrix.
        """

        term_counters = [Counter(text.lower().split()) for text in corpus]

        self._feature_names = dict()
        for counter in term_counters:
            self._feature_names.update(dict.fromkeys(counter))
        self._feature_names = list(self._feature_names)

        count_matrix = []
        for counter in term_counters:
            count_matrix.append([counter[key] for key in self._feature_names])

        return count_matrix

    def get_feature_names(self):
        """
        Return a list of feature names.
        """

        return self._feature_names


if __name__ == '__main__':
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    print('Corpus: ')
    print(*corpus, sep='\n')
    print('Feature names:')
    print(vectorizer.get_feature_names())
    # Out: ['crock', 'pot', 'pasta', 'never', 'boil', 'again', 'pomodoro',
    #       'fresh', 'ingredients', 'parmesan', 'to', 'taste']
    print('Count matrix:')
    print(count_matrix)
    # Out: [[1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    #       [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
