import nltk
from nltk.tokenize import word_tokenize
import json

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Input sentence
sentence = "The quantum entanglement of particles was observed using a superconducting qubit array."

# Tokenize the sentence
tokens = word_tokenize(sentence)

# Get POS tags using the Penn Treebank tagset
tagged_sentence = nltk.pos_tag(tokens)

# Convert to a list of dictionaries
tagged_dicts = [{"word": word, "tag": tag} for word, tag in tagged_sentence]

# Print the result
tagged_json = json.dumps(tagged_dicts, indent=2)
print(tagged_json)
