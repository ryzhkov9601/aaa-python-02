from collections import Counter


class CountVectorizer:
    """
    A class to represent a count vectorizer's functionality.
    """

    def __init__(self):
        self._feature_names = dict()

    def fit_transform(self, corpus):
        """
        Fits for given corpus, then return count matrix.
        """

        term_counters = [Counter(text.lower().split()) for text in corpus]
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
