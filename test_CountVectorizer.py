import pytest
import CountVectorizer


class Test_CountVectorizer:
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]

    feature_names = [
        'crock', 'pot', 'pasta', 'never', 'boil',
        'again', 'pomodoro', 'fresh', 'ingredients',
        'parmesan', 'to', 'taste'
    ]

    count_matrix = [
        [1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    ]

    @pytest.fixture
    def cv(self):
        return CountVectorizer.CountVectorizer()

    @pytest.fixture
    def cv_fitted(self, cv):
        cv.fit_transform(self.corpus)
        return cv

    def test_fit_transform(self, cv):
        count_matrix = cv.fit_transform(self.corpus)
        assert count_matrix == self.count_matrix

    def test_get_feature_names(self, cv_fitted):
        feature_names = cv_fitted.get_feature_names()
        assert feature_names == self.feature_names
