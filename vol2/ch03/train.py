import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.trainer import Trainer
from common.optimizer import Adam
from simple_skip_gram import SimpleSkipGram
from common.util import preprocess, create_contexts_target, convert_one_hot

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = "Dragostea Din Tei è una della tante hit della dance band moldava O-Zone. La loro storia è cominciata nella Repubblica Moldava, dove Dan, Arsenie e Radu hanno conosciuto i loro primi successi musicali. In questo modo si sono convinti che la musica pop rumena poteva essere interessante x quanto riguarda il mercato europeo e potevano esserci i presupposti per l'inizio di una carriera Internazionale. Ecco perché nel 2002 gli O-Zone hanno dato un loro demo a Dan Popi un manager di un'etichetta discografica rumena nel quale hanno riposto la loro fiducia. Il loro primo singolo in Romania è stato: Numai Tu tratto dal loro album Number 1. Numai Tu ha conquistato le classifiche radio e la Top 100 rumena. Nell'autunno 2002 gli O-zone tornano con un nuovo singolo Despre Tine e le nuove stelle della musica dance si accorgono di aver contaminato sia le radio sia le televisioni della Romania con l O-Zone Mania."

corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

model = SimpleSkipGram(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])