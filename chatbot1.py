import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import wikipedia
# nltk.download("wordnet")
# nltk.download("punkt_tab")
# nltk.download("vader_lexicon")
# nltk.download("averaged_perceptron_tagger_eng")

lemmatizer = WordNetLemmatizer()


def lemma_me(sent):
    sentence_token = nltk.word_tokenize(sent.lower())
    pos_tags = nltk.pos_tag(sentence_token)

    sentence_lemmas = []
    for token, pos_tag in zip(sentence_token, pos_tags):
        if pos_tag[1][0].lower() in ["r", "n", "v", "a"]:
            lemma = lemmatizer.lemmatize(token, pos_tag[1][0].lower())
            sentence_lemmas.append(lemma)

    return sentence_lemmas


text = wikipedia.page("Chelsea F.C.").content


def process(texts, questions):
    sentence_tokens = nltk.sent_tokenize(texts)
    sentence_tokens.append(questions)

    tv = TfidfVectorizer(tokenizer=lemma_me)
    tf = tv.fit_transform(sentence_tokens)

    values = cosine_similarity(tf[-1], tf)
    index = values.argsort()[0][-2]
    values_flat = values.flatten()
    values_flat.sort()
    coeff = values_flat[-2]
    if coeff > 0.3:
        return sentence_tokens[index]


while True:
    question = input("What can i help you about?\n")
    output = process(text, question)
    if output:
        print(f"{output}\n")
    elif question == "quit":
        break
    else:
        print("I dont know.")
