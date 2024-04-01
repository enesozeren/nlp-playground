# Description: This file contains the static variables that are used in the ToyNamer project.

turkish_chars = 'ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZabcçdefgğhıijklmnoöprsştuüvyz'
VOCAB_SIZE = len(turkish_chars) + 2 # +2 for start and end of the word character
char_to_int = {char: index for index, char in enumerate(turkish_chars)}
char_to_int['<S>'] = len(turkish_chars)
char_to_int['<E>'] = len(turkish_chars) + 1
int_to_char = {index: char for char, index in char_to_int.items()}