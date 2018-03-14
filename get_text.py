import string,re

text_blob = open("text_blob copy.txt","r")

words = str(text_blob.read()).replace("'","").replace('"', '').replace("(","").replace(")","").replace("\xe2\x80\x98","").replace("\xe2\x80\x99","")\
    .replace("\xe2\x80\x93","")
words_no_num = re.sub(r'[0-9]+', '', words)
words_no_punc = words_no_num.translate(None, string.punctuation)
doc_string = words_no_punc.lower().split()
print doc_string
#remove punctuaction
# for entry in doc_string:

print len(doc_string)

