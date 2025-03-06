# -*- coding: utf-8 -*-
"""326295417_326621927_Targil5.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1t1ZaqAwV8JXrcNswJX_oawVA3h2AeJ8P
"""

!git clone https://github.com/GuyKabiri/language_models

!pip install tensorflow-gpu
!pip install --upgrade tensorflow
!pip install keras-preprocessing-gpu
import tensorflow as tf
!pip install tensorboardX
import importlib
import language_models.util as util  # Import util from the language_models library

importlib.reload(util)  # Reload the module after modification

device_name = tf.test.gpu_device_name()
print(device_name)
# if device_name != '/device:GPU:0':
#   raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))
print("GPU Available:", tf.config.list_physical_devices('GPU'))

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Input, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.losses import sparse_categorical_crossentropy
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorboardX import SummaryWriter
import ipywidgets as widgets
from IPython.display import display

class Args:
  epochs = 20 # Number of epochs
  embedding_size = 300 # Size of the word embeddings on the input layer.
  out_every = 1 # Output every n epochs.
  lr = 0.001 # Learning rate
  batch = 128 # Batch size
  task = 'wikisimple'
  data = './data' # Data file. Should contain one sentence per line.
  lstm_capacity = 256
  max_length = None # Sentence max length.
  top_words = 10000 # Word list size.
  limit = None # Character cap for the corpus - not relevant in our exercise.
  tb_dir = './runs/words' # Tensorboard directory
  seed = -1 # RNG seed. Negative for random (seed is printed for reproducability).
  extra = None # Number of extra LSTM layers.

options = Args()

options = Args()
tbw = SummaryWriter(log_dir=options.tb_dir)

if options.seed < 0:
    seed = random.randint(0, 1000000)
    print('random seed: ', seed)
    np.random.seed(seed)
else:
    np.random.seed(options.seed)

if options.task == 'wikisimple':

    x, w2i, i2w = util.load_words(util.DIR + '/datasets/wikisimple.txt', vocab_size=options.top_words, limit=options.limit)

    # Finding the length of the longest sequence
    x_max_len = max([len(sentence) for sentence in x])

    numwords = len(i2w)
    print('max sequence length ', x_max_len)
    print(numwords, 'distinct words')

    x = util.batch_pad(x, options.batch, add_eos=True)

elif options.task == 'file':

    x, w2i, i2w = util.load_words(options.data_dir, vocab_size=options.top_words, limit=options.limit)

    # Finding the length of the longest sequence
    x_max_len = max([len(sentence) for sentence in x])

    numwords = len(i2w)
    print('max sequence length ', x_max_len)
    print(numwords, 'distinct words')

    x = util.batch_pad(x, options.batch, add_eos=True)

else:
    raise Exception('Task {} not recognized.'.format(options.task))

def decode(seq):
    return ' '.join(i2w[id] for id in seq)

print('Finished data loading. ', sum([b.shape[0] for b in x]), ' sentences loaded')

def sparse_loss(y_true, y_pred):
    return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

def generate_seq(model : Model, seed, size, temperature=1.0):
    """
    :param model: The complete RNN language model
    :param seed: The first few wordas of the sequence to start generating from
    :param size: The total size of the sequence to generate
    :param temperature: This controls how much we follow the probabilities provided by the network. For t=1.0 we just
        sample directly according to the probabilities. Lower temperatures make the high-probability words more likely
        (providing more likely, but slightly boring sentences) and higher temperatures make the lower probabilities more
        likely (resulting is weirder sentences). For temperature=0.0, the generation is _greedy_, i.e. the word with the
        highest probability is always chosen.
    :return: A list of integers representing a samples sentence
    """

    ls = seed.shape[0]

    # Due to the way Keras RNNs work, we feed the model a complete sequence each time. At first it's just the seed,
    # zero-padded to the right length. With each iteration we sample and set the next character.

    tokens = np.concatenate([seed, np.zeros(size - ls)])

    for i in range(ls, size):

        probs = model.predict(tokens[None,:])

        # Extract the i-th probability vector and sample an index from it
        next_token = util.sample_logits(probs[0, i-1, :], temperature=temperature)

        tokens[i] = next_token

    return [int(t) for t in tokens]

#Section 1
train_data, temp_data = train_test_split(x, test_size=0.2, random_state=42)
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

#Section 2
def calculate_perplexity(model, data):
    total_loss = 0
    total_samples = 0

    for batch in data:
        X = batch
        loss_value = model.evaluate(X, X, verbose=0)
        total_loss += loss_value * len(X)
        total_samples += len(X)

    avg_loss = total_loss / total_samples
    perplexity = np.exp(avg_loss)
    return perplexity

#Section 3
class ReverseLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(ReverseLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reverse(inputs, axis=[self.axis])

def create_model(numwords, embedding_size, lstm_capacity, extra_layers, reversed=False):
    input = Input(shape=(None,))

    embedding = Embedding(numwords, embedding_size, input_length=None)(input)

    if reversed:
        embedded = ReverseLayer(axis=1)(embedding)
    else:
        embedded = embedding

    lstm_layer = LSTM(lstm_capacity, return_sequences=True)
    hidden = lstm_layer(embedded)

    if extra_layers > 0:
        for _ in range(extra_layers):
            hidden = LSTM(lstm_capacity, return_sequences=True)(hidden)

    output = TimeDistributed(Dense(numwords, activation='softmax'))(hidden)

    model = tf.keras.Model(inputs=input, outputs=output)

    return model

#Section 5
def get_sentence_probability(model, sentence, w2i, epsilon=1e-9):
    sentence_ids = []

    for word in sentence.split():
        if word not in w2i:
            return 0
        sentence_ids.append(w2i[word])

    sequence = np.array(sentence_ids)
    probs = model.predict(sequence[None, :])
    log_probs = [np.log(probs[0, i, sentence_ids[i]] + epsilon) for i in range(len(sentence_ids))]
    log_probability = np.sum(log_probs)
    return np.exp(log_probability)

#Section 7
def predict_next_word_ui(model, input_word, result_label, temperature=0.0, size=1):
    try:
        sentence_ids = [w2i[word] for word in input_word.split() if word in w2i]
    except KeyError:
        result_label.value = "Word not in vocabulary."
        return

    sequence = np.array(sentence_ids)

    generated_sequence = generate_seq(model, sequence, size=size + 1, temperature=temperature)

    predicted_word_index = generated_sequence[-1]
    predicted_word = i2w[predicted_word_index]
    print(predicted_word)
    result_label.value = f'Predicted word for "{input_word}": {predicted_word}'

def create_ui(model):
    input_word_widget = widgets.Text(
        description="Enter word:",
        placeholder="Enter a word here",
    )

    predict_button_widget = widgets.Button(description="Predict Next Word")
    result_label = widgets.Label(value="")

    predict_button_widget.on_click(lambda b: predict_next_word_ui(model, input_word_widget.value, result_label))

    display(input_word_widget, predict_button_widget, result_label)

#Section 4
models = []
for reversed in [False, True]:
    for extra_layers in [0, 1]:
        model = create_model(numwords=len(i2w), embedding_size=options.embedding_size, lstm_capacity=options.lstm_capacity, extra_layers=extra_layers, reversed=reversed)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=options.lr), loss=sparse_loss)
        models.append(model)
i=-1
for model_type in ["Regular", "Reverse"]:
  for extra_layers in [0, 1]:
      i = i+1
      model = models[i]
      print(f'{model_type} train with ({extra_layers + 1}) LSTM layers')
      epoch = 0
      instances_seen = 0
      while epoch < options.epochs:
          for batch in tqdm(train_data):
              n, l = batch.shape
              batch_shifted = np.concatenate([np.ones((n, 1)), batch], axis=1)
              batch_out = np.concatenate([batch, np.zeros((n, 1))], axis=1)

              loss = model.train_on_batch(batch_shifted, batch_out[:, :, None])
              instances_seen += n
              tbw.add_scalar('lm/batch-loss', float(loss), instances_seen)
          print(loss)
          epoch += 1

      print("\n\n")
      #Section 8
      for data_set in ['train', 'validation', 'test']:
        data = locals()[f'{data_set}_data']
        perplexity = calculate_perplexity(model, data)
        print(f'Perplexity of {model_type} LSTM ({extra_layers + 1}) on {data_set} data: {perplexity}')

      print("\n\n")
      #Section 6, 9
      for temp in [0.1, 1, 10]:
          print('\n### TEMP ', temp)
          seed = [w2i[word] for word in "I love".split()]
          seed = np.insert(seed, 0, 1)
          gen = generate_seq(model, seed, 8, temperature=temp)
          sentence = decode(seed[1:]) + " " + decode(gen[len(seed):])
          print(sentence)
          prob = get_sentence_probability(model, sentence, w2i)
          print(f'Probability of this sentence: {prob}')

      sentence = "I love cupcakes"
      prob = get_sentence_probability(model, sentence, w2i)
      print(f'\nProbability of "{sentence}": {prob}')

      print("\n\n\n")

#Section 7
model = models[0]
create_ui(model)

model = models[1]
create_ui(model)

model = models[2]
create_ui(model)

model = models[3]
create_ui(model)