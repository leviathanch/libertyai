import nltk
import spacy

nlp = spacy.load("en_core_web_sm")

with open('/home/leviathan/libertyai/critique_of_interventionism.txt','r') as f:
    text = f.read()
    f.close()


with open('/home/leviathan/libertyai/critique_of_interventionism_clean.txt','w') as f:
    texts = text.split('\n\n')
    for t in texts:
        string_unicode = t.replace('\n',' ').replace('-',' ')
        string_encode = string_unicode.encode("ascii", "ignore")
        string_decode = string_encode.decode()
        doc = nlp(string_decode)
        for sent in doc.sents:
            if sent[0].is_title and sent[-1].is_punct:
                has_noun = 2
                has_verb = 1
                for token in sent:
                    if token.pos_ in ["NOUN", "PROPN", "PRON"]:
                        has_noun -= 1
                    elif token.pos_ == "VERB":
                        has_verb -= 1
                if has_noun < 1 and has_verb < 1:
                    f.write(str(sent))
                    f.write('\n\n')
    f.close()
