import gensim
from gensim import corpora
from gensim.corpora import Dictionary, bleicorpus
from gensim.models import ldamodel
import pyLDAvis

#Como cargar el corpus y su respectivo diccionario en python.
corpus = bleicorpus.BleiCorpus('Tarea3/corpus_lda/corpus_lda.lda_c')
dictionary = Dictionary.load('Tarea3/corpus_lda/corpus_lda.dict')

#pregunta 3
lda = ldamodel.LdaModel(corpus, num_topics=3)
lda.save("res_lda/lda_3/lda_3")

lda = ldamodel.LdaModel(corpus, num_topics=5)
lda.save("res_lda/lda_5/lda_5")

lda = ldamodel.LdaModel(corpus, num_topics=10)
lda.save("res_lda/lda_10/lda_10")

