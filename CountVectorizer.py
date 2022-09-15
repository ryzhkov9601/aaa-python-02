from collections import Counter


class CountVectorizer:
    def __init__(self):
        pass

    def fit_transform(self, corpus):
        self._feature_names = dict()

        term_counters = [Counter(text.lower().split()) for text in corpus]
        for counter in term_counters:
            self._feature_names.update(dict.fromkeys(counter))
        self._feature_names = list(self._feature_names)

        result = []
        for counter in term_counters:
            result.append([counter[key] for key in self._feature_names])

        return result

    def get_feature_names(self):
        return self._feature_names
