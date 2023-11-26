import gensim
import gensim.models.word2vec as w2v
from nltk.tokenize import word_tokenize

import chromadb


##create collection in chromadb
chroma_client = chromadb.PersistentClient(path="chroma.db")
collection = chroma_client.create_collection("Talkpdf")



