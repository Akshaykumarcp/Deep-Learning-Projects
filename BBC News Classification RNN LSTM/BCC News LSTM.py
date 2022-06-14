import tensorflow as tf
print(tf.__version__)
# 2.9.1

import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
STOPWORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

# Put the hyparameters at the top like this to make it easier to change and edit.

vocab_size = 5000
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8

# First, let's define two lists that containing articles and labels. In the meantime, we remove stopwords.

articles = []
labels = []

with open("bbc-text.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[0])
        article = row[1]
        for word in STOPWORDS:
            token = ' ' + word + ' '
            article = article.replace(token, ' ')
            article = article.replace(' ', ' ')
        articles.append(article)

print(len(labels))
print(len(articles))

""" 
2225
2225 """

# There are only 2,225 articles in the data. Then we split into training set and 
# validation set, according to the parameter we set earlier, 80% for training, 20% for validation.

train_size = int(len(articles) * training_portion) # 1780

train_articles = articles[0: train_size]
train_labels = labels[0: train_size]

validation_articles = articles[train_size:]
validation_labels = labels[train_size:]

print(train_size)
print(len(train_articles))
print(len(train_labels))
print(len(validation_articles))
print(len(validation_labels))

""" 
1780
1780
1780
445
445 """

# Tokenizer does all the heavy lifting for us. In our articles that it was tokenizing, 
# it will take 5,000 most common words. oov_token is to put a special value in when an 
# unseen word is encountered. This means I want "OOV" in bracket to be used to for words 
# that are not in the word index. "fit_on_text" will go through all the text and create dictionary like this:

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index

# You can see that "OOV" in bracket is number 1, "said" is number 2, "mr" is number 3, and so on.

dict(list(word_index.items())[0:10])
""" 
{'<OOV>': 1,
 'said': 2,
 'mr': 3,
 'would': 4,
 'year': 5,
 'also': 6,
 'people': 7,
 'new': 8,
 'us': 9,
 'one': 10} """

""" This process cleans up our text, lowercase, and remove punctuations.

After tokenization, the next step is to turn thoes tokens into lists of sequence. """

train_sequences = tokenizer.texts_to_sequences(train_articles)

# This is the 11th article in the training data that has been turned into sequences.

print(train_sequences[10])
""" 
[2522, 1, 313, 1, 77, 743, 690, 40, 24, 313, 1, 1, 1, 9, 1758, 1, 1, 11, 2522, 77, 668, 1, 1, 226, 371, 1, 226, 371, 2, 897, 2, 922, 764, 2398, 1, 1246, 1787, 1, 7, 94, 1814, 1, 1, 1, 1, 1, 1, 4816, 2, 1, 1, 208, 4596, 1, 4, 16, 10, 2965, 2, 1603, 449, 4817, 2, 1, 121, 22, 437, 1, 449, 28, 2265, 4044, 5, 106, 77, 3879, 1, 2, 1, 1, 1, 3, 645, 1, 1, 2, 1, 934, 733, 3, 2458, 444, 28, 4818, 1, 463, 77, 1, 34, 889, 2459, 7, 1, 4383, 224, 39, 2, 1, 3, 3750, 784, 3618, 1, 77, 1, 63, 514, 2, 922, 764, 1, 169, 53, 735, 1, 313, 1, 1, 702, 2, 1, 19, 1787, 8, 19, 1121, 1, 11, 1, 908, 1959, 201, 2, 1, 7, 1, 1, 3, 3065, 77, 1, 181, 371, 18, 1, 3, 1703, 1, 645, 25, 595, 1, 1544, 4819, 881, 1421, 7, 1, 1956, 39, 2, 93, 743, 415, 1, 134, 581, 11, 668, 396, 1604, 77, 582, 2, 1, 3, 1, 1760, 1, 898, 1, 3158, 1, 1466, 9, 33, 1, 11, 2522, 668, 77, 3062, 4813, 1, 1, 44, 1, 1, 41, 1, 25, 949, 104, 1920, 12, 777, 391, 81, 1079, 1, 28, 982, 77, 459, 77, 53, 396, 1604, 1444, 473, 68, 71, 135, 6, 983, 1197, 4384, 337]

When we train neural networks for NLP, we need sequences to be in the same size, that's why we use padding. 
Our max_length is 200, so we use pad_sequences to make all of our articles the same length which is 200 in my example. 
That's why you see that the 1st article was 426 in length, becomes 200, the 2nd article was 192 in length, becomes 200, and so on.
"""

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(len(train_sequences[0]))
print(len(train_padded[0]))

print(len(train_sequences[1]))
print(len(train_padded[1]))

print(len(train_sequences[10]))
print(len(train_padded[10]))

""" 
425
200
192
200
186
200 """

""" In addtion, there is padding type and truncating type, there are all "post". Means for example, 
for the 11th article, it was 186 in length, we padded to 200, and we padded at the end, add 14 zeros. """

print(train_sequences[10])
# [2431, 1, 225, 4996, 22, 642, 587, 225, 4996, 1, 1, 1663, 1, 1, 2431, 22, 565, 1, 1, 140, 278, 1, 140, 278, 796, 823, 662, 2307, 1, 1144, 1694, 1, 1721, 4997, 1, 1, 1, 1, 1, 4738, 1, 1, 122, 4514, 1, 2, 2874, 1505, 352, 4739, 1, 52, 341, 1, 352, 2172, 3962, 41, 22, 3795, 1, 1, 1, 1, 543, 1, 1, 1, 835, 631, 2366, 347, 4740, 1, 365, 22, 1, 787, 2367, 1, 4302, 138, 10, 1, 3666, 682, 3531, 1, 22, 1, 414, 823, 662, 1, 90, 13, 633, 1, 225, 4996, 1, 599, 1, 1694, 1021, 1, 4998, 808, 1864, 117, 1, 1, 1, 2974, 22, 1, 99, 278, 1, 1608, 4999, 543, 493, 1, 1443, 4741, 778, 1320, 1, 1861, 10, 33, 642, 319, 1, 62, 478, 565, 301, 1506, 22, 479, 1, 1, 1666, 1, 797, 1, 3066, 1, 1365, 6, 1, 2431, 565, 22, 2971, 4735, 1, 1, 1, 1, 1, 850, 39, 1825, 675, 297, 26, 979, 1, 882, 22, 361, 22, 13, 301, 1506, 1343, 374, 20, 63, 883, 1096, 4303, 247]
print(train_padded[10])
""" [2431    1  225 4996   22  642  587  225 4996    1    1 1663    1    1
 2431   22  565    1    1  140  278    1  140  278  796  823  662 2307
    1 1144 1694    1 1721 4997    1    1    1    1    1 4738    1    1
  122 4514    1    2 2874 1505  352 4739    1   52  341    1  352 2172
 3962   41   22 3795    1    1    1    1  543    1    1    1  835  631
 2366  347 4740    1  365   22    1  787 2367    1 4302  138   10    1
 3666  682 3531    1   22    1  414  823  662    1   90   13  633    1
  225 4996    1  599    1 1694 1021    1 4998  808 1864  117    1    1
    1 2974   22    1   99  278    1 1608 4999  543  493    1 1443 4741
  778 1320    1 1861   10   33  642  319    1   62  478  565  301 1506
   22  479    1    1 1666    1  797    1 3066    1 1365    6    1 2431
  565   22 2971 4735    1    1    1    1    1  850   39 1825  675  297
   26  979    1  882   22  361   22   13  301 1506 1343  374   20   63
  883 1096 4303  247    0    0    0    0    0    0    0    0    0    0
    0    0    0    0] """

# And for the 1st article, it was 426 in length, we truncated to 200, and we truncated at the end.

print(train_sequences[0])
# [91, 160, 1141, 1106, 49, 979, 755, 1, 89, 1304, 4289, 129, 175, 3654, 1214, 1195, 1578, 42, 7, 893, 91, 1, 334, 85, 20, 14, 130, 3262, 1215, 2421, 570, 451, 1376, 58, 3378, 3521, 1661, 8, 921, 730, 10, 844, 1, 9, 598, 1579, 1107, 395, 1941, 1106, 731, 49, 538, 1398, 2012, 1623, 134, 249, 113, 2355, 795, 4981, 980, 584, 10, 3957, 3958, 921, 2562, 129, 344, 175, 3654, 1, 1, 39, 62, 2867, 28, 9, 4723, 18, 1305, 136, 416, 7, 143, 1423, 71, 4501, 436, 4982, 91, 1107, 77, 1, 82, 2013, 53, 1, 91, 6, 1008, 609, 89, 1304, 91, 1964, 131, 137, 420, 9, 2868, 38, 152, 1234, 89, 1304, 4724, 7, 436, 4982, 3154, 6, 2492, 1, 431, 1126, 1, 1424, 571, 1261, 1902, 1, 766, 9, 538, 1398, 2012, 134, 2069, 400, 845, 1965, 1601, 34, 1717, 2869, 1, 1, 2422, 244, 9, 2624, 82, 732, 6, 1173, 1196, 152, 720, 591, 1, 124, 28, 1305, 1690, 432, 83, 933, 115, 20, 14, 18, 3155, 1, 37, 1484, 1, 23, 37, 87, 335, 2356, 37, 467, 255, 1965, 1359, 328, 1, 299, 732, 1174, 18, 2870, 1717, 1, 294, 756, 1074, 395, 2014, 387, 431, 2014, 2, 1360, 1, 1717, 2166, 67, 1, 1, 1718, 249, 1662, 3059, 1175, 395, 41, 878, 246, 2792, 345, 53, 548, 400, 2, 1, 1, 655, 1361, 203, 91, 3959, 91, 90, 42, 7, 320, 395, 77, 893, 1, 91, 1106, 400, 538, 9, 845, 2422, 11, 38, 1, 995, 514, 483, 2070, 160, 572, 1, 128, 7, 320, 77, 893, 1216, 1126, 1463, 346, 54, 2214, 1217, 741, 92, 256, 274, 1019, 71, 623, 346, 2423, 756, 1215, 2357, 1719, 1, 3784, 3522, 1, 1126, 2014, 177, 371, 1399, 77, 53, 548, 105, 1141, 3, 1, 1047, 93, 2962, 1, 2625, 1, 102, 902, 440, 452, 2, 3, 1, 2871, 451, 1425, 43, 77, 429, 31, 8, 1019, 921, 1, 2562, 30, 1, 91, 1691, 879, 89, 1304, 91, 1964, 1, 30, 8, 1624, 1, 1, 4290, 1580, 4289, 656, 1, 3785, 1008, 572, 4291, 2867, 10, 880, 656, 58, 1, 1262, 1, 1, 91, 1554, 934, 4723, 1, 577, 4106, 10, 9, 235, 2012, 91, 134, 1, 95, 656, 3263, 1, 58, 520, 673, 2626, 3785, 4983, 3379, 483, 4725, 39, 4501, 1, 91, 1748, 673, 269, 116, 239, 2627, 354, 644, 58, 4107, 757, 3655, 4723, 146, 1, 400, 7, 71, 1749, 1107, 767, 910, 118, 584, 3380, 1316, 1579, 1, 1602, 7, 893, 77, 77]
print(train_padded[0])
""" [  91  160 1141 1106   49  979  755    1   89 1304 4289  129  175 3654
 1214 1195 1578   42    7  893   91    1  334   85   20   14  130 3262
 1215 2421  570  451 1376   58 3378 3521 1661    8  921  730   10  844
    1    9  598 1579 1107  395 1941 1106  731   49  538 1398 2012 1623
  134  249  113 2355  795 4981  980  584   10 3957 3958  921 2562  129
  344  175 3654    1    1   39   62 2867   28    9 4723   18 1305  136
  416    7  143 1423   71 4501  436 4982   91 1107   77    1   82 2013
   53    1   91    6 1008  609   89 1304   91 1964  131  137  420    9
 2868   38  152 1234   89 1304 4724    7  436 4982 3154    6 2492    1
  431 1126    1 1424  571 1261 1902    1  766    9  538 1398 2012  134
 2069  400  845 1965 1601   34 1717 2869    1    1 2422  244    9 2624
   82  732    6 1173 1196  152  720  591    1  124   28 1305 1690  432
   83  933  115   20   14   18 3155    1   37 1484    1   23   37   87
  335 2356   37  467  255 1965 1359  328    1  299  732 1174   18 2870
 1717    1  294  756] """

# Then we do the same for the validation sequences. Note that we should expect more out of vocabulary words from validation articles because word index were derived from the training articles.

validation_sequences = tokenizer.texts_to_sequences(validation_articles)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(len(validation_sequences))
print(validation_padded.shape)

""" 
445
(445, 200)

Now we are going to look at the labels. because our labels are text, so we will tokenize them, 
when training, labels are expected to be numpy arrays. So we will turn list of labels into numpy arrays like so:
 """

print(set(labels))
# {'politics', 'entertainment', 'tech', 'sport', 'business'}

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

print(training_label_seq[0])
print(training_label_seq[1])
print(training_label_seq[2])
print(training_label_seq.shape)

print(validation_label_seq[0])
print(validation_label_seq[1])
print(validation_label_seq[2])
print(validation_label_seq.shape)

""" 
[4]
[2]
[1]
(1780, 1)
[5]
[4]
[3]
(445, 1) """

# Before training deep neural network, we want to explore what our original article and article after padding look like. 
# Running the following code, we explore the 11th article, we can see that some words become "OOV", 
# because they did not make to the top 5,000.

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_article(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_article(train_padded[10]))
print('---')
print(train_articles[10])

""" 
berlin <OOV> anti nazi film german movie anti nazi <OOV> <OOV> drawn <OOV> <OOV> berlin film festival <OOV> <OOV> final days <OOV> final days member white rose movement <OOV> 21 arrested <OOV> brother hans <OOV> <OOV> <OOV> <OOV> <OOV> tyranny <OOV> <OOV> director marc <OOV> said feeling responsibility keep legacy <OOV> going must <OOV> keep ideas alive added film drew <OOV> <OOV> <OOV> <OOV> trial <OOV> <OOV> <OOV> east germany secret police discovery <OOV> behind film <OOV> worked closely <OOV> relatives including one <OOV> sisters ensure historical <OOV> film <OOV> members white rose <OOV> group first started <OOV> anti nazi <OOV> summer <OOV> arrested dropped <OOV> munich university calling day <OOV> <OOV> <OOV> regime film <OOV> six days <OOV> arrest intense trial saw <OOV> initially deny charges ended <OOV> appearance one three german films <OOV> top prize festival south african film version <OOV> <OOV> opera <OOV> shot <OOV> town <OOV> language also <OOV> berlin festival film entitled u <OOV> <OOV> <OOV> <OOV> <OOV> story set performed 40 strong music theatre <OOV> debut film performance film first south african feature 25 years second nominated golden bear award ? ? ? ? ? ? ? ? ? ? ? ? ? ?
---
berlin cheers anti-nazi film german movie anti-nazi resistance heroine drawn loud applause berlin film festival.  sophie scholl - final days portrays final days member white rose movement. scholl  21  arrested beheaded brother  hans  1943 distributing leaflets condemning  abhorrent tyranny  adolf hitler. director marc rothemund said:  feeling responsibility keep legacy scholls going.   must somehow keep ideas alive   added.  film drew transcripts gestapo interrogations scholl trial preserved archive communist east germany secret police. discovery inspiration behind film rothemund  worked closely surviving relatives  including one scholl sisters  ensure historical accuracy film. scholl members white rose resistance group first started distributing anti-nazi leaflets summer 1942. arrested dropped leaflets munich university calling  day reckoning  adolf hitler regime. film focuses six days scholl arrest intense trial saw scholl initially deny charges ended defiant appearance. one three german films vying top prize festival.  south african film version bizet tragic opera carmen shot cape town xhosa language also premiered berlin festival. film entitled u-carmen ekhayelitsha carmen khayelitsha township story set. performed 40-strong music theatre troupe debut film performance. film first south african feature 25 years second nominated golden bear award.
Now we can implement LSTM. Here is my code that I build a tf.keras.Sequential model and start with an embedding layer. An embedding layer stores one vector per word. When called, it converts the sequences of word indices into sequences of vectors. After training, words with similar meanings often have the similar vectors.

Next is how to implement LSTM in code. The Bidirectional wrapper is used with a LSTM layer, this propagates the input forwards and backwards through the LSTM layer and then concatenates the outputs. This helps LSTM to learn long term dependencies. We then fit it to a dense neural network to do classification.

This index-lookup is much more efficient than the equivalent operation of passing a one-hot encoded vector through a tf.keras.layers.Dense layer.
"""

model = tf.keras.Sequential([
    # Add an Embedding layer expecting input vocab of size 5000, and output embedding dimension of size 64 we set at the top
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
#    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    # use ReLU in place of tanh function since they are very good alternatives of each other.
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    # Add a Dense layer with 6 units and softmax activation.
    # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
    tf.keras.layers.Dense(6, activation='softmax')
])
model.summary()

""" Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_3 (Embedding)      (None, None, 64)          320000    
_________________________________________________________________
bidirectional_3 (Bidirection (None, 128)               66048     
_________________________________________________________________
dense_6 (Dense)              (None, 64)                8256      
_________________________________________________________________
dense_7 (Dense)              (None, 6)                 390       
=================================================================
Total params: 394,694
Trainable params: 394,694
Non-trainable params: 0
_________________________________________________________________
In our model summay, we have our embeddings, our Bidirectional contains LSTM, followed by two dense layers. The output from Bidirectional is 128, because it doubled what we put in LSTM. We can also stack LSTM layer but I found the results worse.
 """
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
num_epochs = 10
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)

""" 
Epoch 1/10
56/56 - 7s - loss: 1.6063 - accuracy: 0.2921 - val_loss: 1.4250 - val_accuracy: 0.4112 - 7s/epoch - 120ms/step
Epoch 2/10
56/56 - 4s - loss: 0.9994 - accuracy: 0.6124 - val_loss: 0.8004 - val_accuracy: 0.6831 - 4s/epoch - 67ms/step
Epoch 3/10
56/56 - 3s - loss: 0.4084 - accuracy: 0.8674 - val_loss: 0.4837 - val_accuracy: 0.8427 - 3s/epoch - 59ms/step
Epoch 4/10
56/56 - 3s - loss: 0.1525 - accuracy: 0.9579 - val_loss: 0.2374 - val_accuracy: 0.9303 - 3s/epoch - 55ms/step
Epoch 5/10
56/56 - 3s - loss: 0.0545 - accuracy: 0.9888 - val_loss: 0.2229 - val_accuracy: 0.9416 - 3s/epoch - 55ms/step
Epoch 6/10
56/56 - 3s - loss: 0.0261 - accuracy: 0.9944 - val_loss: 0.2169 - val_accuracy: 0.9393 - 3s/epoch - 58ms/step
Epoch 7/10
56/56 - 3s - loss: 0.0075 - accuracy: 0.9983 - val_loss: 0.2686 - val_accuracy: 0.9438 - 3s/epoch - 55ms/step
Epoch 8/10
56/56 - 3s - loss: 0.0058 - accuracy: 0.9983 - val_loss: 0.2907 - val_accuracy: 0.9371 - 3s/epoch - 57ms/step
Epoch 9/10
56/56 - 3s - loss: 0.0029 - accuracy: 0.9994 - val_loss: 0.3033 - val_accuracy: 0.9348 - 3s/epoch - 55ms/step
Epoch 10/10
56/56 - 3s - loss: 0.0088 - accuracy: 0.9966 - val_loss: 0.3731 - val_accuracy: 0.9101 - 3s/epoch - 56ms/step """

from matplotlib import pyplot as plt

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

# PREDICTION

txt = ["A WeWork shareholder has taken the company to court over the near-$1.7bn (£1.3bn) leaving package approved for ousted co-founder Adam Neumann."]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_length)
pred = model.predict(padded)

pred
""" array([[1.4234246e-05, 9.6381044e-01, 3.4862484e-03, 3.0783178e-02,
        2.5258851e-04, 1.6533067e-03]], dtype=float32) """

txt = ["A WeWork shareholder has taken the company to court over the near-$1.7bn (£1.3bn) leaving package approved for ousted co-founder Adam Neumann."]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_length)
pred = model.predict(padded)

labels = ['sport', 'bussiness', 'politics', 'tech', 'entertainment','unknown']

print(pred, labels[np.argmax(pred)])
""" [[1.4234246e-05 9.6381044e-01 3.4862484e-03 3.0783178e-02 2.5258851e-04
  1.6533067e-03]] bussiness """

np.argmax(pred)
# 1