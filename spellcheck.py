from symspellpy import SymSpell
from symspellpy.symspellpy import Verbosity
from itertools import islice

sym_spell_name_words = SymSpell()
sym_spell_name_words.load_dictionary('name-words.txt', 0, 1, separator="$")
sym_spell_name_words.load_dictionary('name-full.txt', 0, 1, separator="$")
# sym_spell_name_words.load_bigram_dictionary('name-bigrams.txt', 0, 2, separator="$")


# results = sym_spell_name_full.lookup_compound('Scvening osze', max_edit_distance=2)
results = sym_spell_name_words.lookup_compound('tegeri mastor of ime', max_edit_distance=2)

for result in results:
    print(result)
