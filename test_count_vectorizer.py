import pytest
import count_vectorizer


class Test_CountVectorizer:
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]

    feature_names = [
        'again', 'boil', 'crock', 'fresh', 'ingredients',
        'never', 'parmesan', 'pasta', 'pomodoro',
        'pot', 'taste', 'to'
    ]

    count_matrix = [
        [1, 1, 1, 0, 0, 1, 0, 2, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    ]

    @pytest.fixture
    def vectorizer(self):
        return count_vectorizer.CountVectorizer()

    @pytest.fixture
    def vectorizer_fitted(self, vectorizer):
        vectorizer.fit_transform(self.corpus)
        return vectorizer

    def test_fit_transform(self, vectorizer):
        count_matrix = vectorizer.fit_transform(self.corpus)
        assert count_matrix == self.count_matrix

    def test_get_feature_names(self, vectorizer_fitted):
        feature_names = vectorizer_fitted.get_feature_names()
        assert feature_names == self.feature_names
