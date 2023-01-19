import spacy
import pandas as pd

file = "train.csv"
df = pd.read_csv(file)

nlp = spacy.load('en_core_web_trf')


def get_ner(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


df['ner'] = df['text'].apply(get_ner)

df.to_csv('train_ner.csv', index=False)
print("NER data generated")
