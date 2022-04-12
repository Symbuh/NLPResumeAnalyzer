# This is going to be starting from scratch
import spacy
import pickle
import random


# Need to get about 200 + resumes of sample data to train with

# Hopefully this program doesn't actually
# return a string String parsing this would be assoluetely terrible

# We need to first take in our PDF resume and turn it into text format.

# He's just following this spacy guide right in front of me cmon man

train_data = pickle.load(open('training_data.pkl', 'rb'))

nlp = spacy.blank('en')

def train_model(train_data):
  if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe()
    nlp.add_pipe(ner, last = True)

  for _, annotations in train_data:
    for ent in annotations.get('entities'):
      ner.add_label(ent[2])

  # get names of other pipes to disable them during training
  other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
  with nlp.disable_pipes(*other_pipes):  # only train NER
    for itn in range(10):
      random.shuffle(train_data)
      losses = {}
      for text, annotations in train_data:
        nlp.update(
          [text],  # batch of texts
          [annotations],  # batch of annotations
          drop=0.3,  # dropout - make it harder to memorise data (try 0.2)
          sgd=optimizer,  # callable to update weights
          losses=losses)
      print(losses)


train_model(train_data)

# Preconfigured Spacy model
nlp_model = spacy.load('nlp_model')

doc = nlp_model(train_data[0][0])

# Prints a demonstration of output
for ent in doc.ents:
  print(ent.text, ent.label_)

# Once we train and save our model, we can use it on our own resume


# Importing and parsing our own resume
import sys, fitz
import PyMuPDF

fname = 'Nicholas Sabadicci Resume.pdf'
doc = fitz.open(fname)
text = ''

for page in doc:
  text += page.getText() + str(page.getText())

print(text)

tx = ' '.join(text.split('\n'))
print(text)

doc = nlp_model(tx)
for ent in doc.ents:
  print(f'{ent.label.upper():{30}}-{ent.text}')

