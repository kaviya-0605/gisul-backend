# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# model = SentenceTransformer(
#     "all-MiniLM-L6-v2"
# )

# topics = [
#     "Biology",
#     "Physics",
#     "Chemistry",
#     "Math",
#     "Computer Science",
#     "General Science"
# ]

# topic_embeddings = model.encode(topics)


# def get_topic(question):

#     q_embedding = model.encode([question])

#     scores = cosine_similarity(
#         q_embedding,
#         topic_embeddings
#     )[0]

#     return topics[np.argmax(scores)]









# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# model = SentenceTransformer(
#     "all-MiniLM-L6-v2"
# )

# TOPICS = {
#     "Biology":
#         "photosynthesis cell dna plant organism gene enzyme protein",

#     "Physics":
#         "force gravity motion velocity acceleration energy",

#     "Chemistry":
#         "acid base reaction molecule element chemical",

#     "Math":
#         "equation algebra geometry triangle calculus",

#     "Computer Science":
#         "computer programming algorithm database software",

#     "General Science":
#         "science experiment observation research"
# }

# topic_names = list(TOPICS.keys())

# topic_embeddings = model.encode(
#     list(TOPICS.values())
# )


# def get_topic(question):

#     q_embedding = model.encode([question])

#     scores = cosine_similarity(
#         q_embedding,
#         topic_embeddings
#     )[0]

#     return topic_names[np.argmax(scores)]






from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

TOPICS = {
    "Biology":
        "photosynthesis dna cell plant gene organism enzyme",

    "Physics":
        "force gravity motion energy velocity acceleration",

    "Chemistry":
        "acid base molecule reaction element compound",

    "Math":
        "algebra geometry equation calculus probability",

    "Computer Science":
        "react node mongodb express javascript python java api backend frontend full stack database programming software algorithm",

    "General Science":
        "science experiment observation research"
}

topic_names = list(TOPICS.keys())

topic_embeddings = model.encode(
    list(TOPICS.values())
)

def get_topic(question):

    q_embedding = model.encode([question])

    scores = cosine_similarity(
        q_embedding,
        topic_embeddings
    )[0]

    return topic_names[np.argmax(scores)]