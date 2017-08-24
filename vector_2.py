import numpy as np

# set to numpy array, loading all into mem to do parallel work
ratings = np.array([
    5,
    2,
    3,
    3,
    4,
    5,
    5,
    1,
    5,
    1,
    3,
    4
])
print('Before mult: ', ratings)

ratings = ratings * 2

print('After mult: ', ratings)
