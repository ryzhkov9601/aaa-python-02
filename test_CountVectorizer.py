import CountVectorizer

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


def test_fit_transform():
    cv = CountVectorizer.CountVectorizer()
    assert cv.fit_transform(corpus) == count_matrix


def test_get_feature_names():
    cv = CountVectorizer.CountVectorizer()
    cv.fit_transform(corpus)
    assert cv.get_feature_names() == feature_names
