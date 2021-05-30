import meilisearch
import json

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

client = meilisearch.Client('http://127.0.0.1:7700')
index = client.index('cards')
with open('zz.json') as f:
  data = json.load(f)

  count = 0
  for chunk in list(chunks(data, 5000)):
      for idx, entry in enumerate(chunk):
          if entry['digital'] == True:
            del chunk[idx]
      index.add_documents(chunk) # => { "updateId": 0 }
      count = count + len(chunk)
      print(f'added: {count}')
#   for entry in data:
#       print(entry)
#       break

# An index is where the documents are stored.
# index = client.index('books')



#     documents = [
#     { 'book_id': 123,  'title': 'Pride and Prejudice' },
#     { 'book_id': 456,  'title': 'Le Petit Prince' },
#     { 'book_id': 1,    'title': 'Alice In Wonderland' },
#     { 'book_id': 1344, 'title': 'The Hobbit' },
#     { 'book_id': 4,    'title': 'Harry Potter and the Half-Blood Prince' },
#     { 'book_id': 42,   'title': 'The Hitchhiker\'s Guide to the Galaxy' }
#     ]

#     # If the index 'books' does not exist, MeiliSearch creates it when you first add the documents.
#     index.add_documents(documents) # => { "updateId": 0 }