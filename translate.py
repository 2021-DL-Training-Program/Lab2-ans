MAX_LENGTH = 22

# build index to word table
index_table = {0:'SOS', 1:'EOS', 2:'PAD'}
for i in range(97, 97+26):
    index_table[i - 94] = chr(i)

# build word to index table
word_table = {}
for i in range(len(index_table)):
    word_table[index_table[i]] = i

def index2word(index):
    word = ''
    for i in range(1, len(index) - 1):
        idx = index[i]
        if idx == 1:
            # EOS(End of string)
            break
        word += index_table[idx]
    return word

def word2index(word):
    index = []
    # SOS
    index.append(0)
    
    for i in range(len(word)):
        index.append(word_table[word[i]])
        
    # EOS
    index.append(1)
    
    # Padding
    while len(index) < MAX_LENGTH:
        index.append(2)
        
    return index
        
