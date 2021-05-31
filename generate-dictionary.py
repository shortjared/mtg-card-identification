import json

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

namewords = open('name-words.txt', 'w')
namebigrams = open('name-bigrams.txt', 'w')
namefull = open('name-full.txt', 'w')

wordFreqs = {}
nameFreqs = {}
wordBigrams = {}

with open('zz.json') as f:
  data = json.load(f)

  count = 0
  for entry in data:
      name = entry['name']
      words = name.split(' ')

      idx = 1
      while idx < len(words):
        a = words[idx-1]
        b = words[idx]
        bigram = f'{a}${b}'
        if bigram in wordBigrams:
          wordBigrams[bigram] += 1
        else:
          wordBigrams[bigram] = 1
        idx += 1

      for word in words:
        if word in wordFreqs:
          wordFreqs[word] += 1
        else:
          wordFreqs[word] = 1
    
      if name in nameFreqs:
        nameFreqs[name] += 1
      else:
        nameFreqs[name] = 1

ordered =  dict(sorted(nameFreqs.items(), key=lambda item: item[1], reverse=True))
for entry in ordered:
  namefull.write(f'{entry}${ordered[entry]}\n')

ordered =  dict(sorted(wordFreqs.items(), key=lambda item: item[1], reverse=True))
for entry in ordered:
  namewords.write(f'{entry}${ordered[entry]}\n')

ordered =  dict(sorted(wordBigrams.items(), key=lambda item: item[1], reverse=True))
for entry in ordered:
  namebigrams.write(f'{entry}${ordered[entry]}\n')

namebigrams.close()
namewords.close()
namefull.close()