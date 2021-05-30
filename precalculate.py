  
import pickle
import mtg


card_detector = mtg.MagicCardDetector('out')
card_detector.read_and_adjust_reference_images('./cards/')

hlist = []
for image in card_detector.reference_images:
    image.original = None
    image.clahe = None
    image.adjusted = None
    hlist.append(image)

with open('all.dat', 'wb') as f:
    pickle.dump(hlist, f)