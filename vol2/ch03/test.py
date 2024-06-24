import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from common.util import preprocess, create_contexts_target, convert_one_hot

text = 'You way goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

contexts, target = create_contexts_target(corpus, window_size=1)

vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
print(target)