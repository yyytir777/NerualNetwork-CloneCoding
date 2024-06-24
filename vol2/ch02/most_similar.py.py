import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from common.util import preprocess, create_co_matrix, most_similar

text = 'You say goodbye and I say hello'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

tmp = most_similar('you', word_to_id, id_to_word, C)
print(tmp)