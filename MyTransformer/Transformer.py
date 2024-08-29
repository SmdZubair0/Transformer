import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, Input, Layer, Dropout
from keras.layers import LayerNormalization, MultiHeadAttention
from tensorflow.keras.preprocessing.text import Tokenizer



class PositionalEncoding(Layer):
    '''
    Positional encoding model

    Constructor takes the following:
    - embedding_dim: the size of the embedding vector
    - seq_len: length of input sequence (default 100)

    Call takes the following:
    - input: the original word embedding (as batch of sentences)

    Call returns the following:
    - output: the word embedding with positional encoding (as batch of sentences)

    '''
    def __init__(self, embedding_dim, seq_len = 100):
        super(PositionalEncoding, self).__init__()

        self.seq_len = seq_len
        self.embedding_dim = embedding_dim


        pos = np.arange(0, seq_len).reshape(seq_len, 1)  # indexes of words

        embed_pos = np.arange(0, embedding_dim)  # indexes of embedding dimensions
        div_term = np.power(10000, 2 * (embed_pos // 2) / embedding_dim)  # division term

        self.positional_encoding = np.zeros((seq_len, embedding_dim))  # positional encoding matrix

        # for i in range(seq_len):
        #     for j in range(embedding_dim):
        #         if j % 2 == 0:
        #             self.positional_encoding[i, j] = np.sin(pos[i] / div_term[j])
        #         else:
        #             self.positional_encoding[i, j] = np.cos(pos[i] / div_term[j])


        self.positional_encoding[:, 0::2] = np.sin(pos / div_term[::2])  # even indices
        self.positional_encoding[:, 1::2] = np.cos(pos / div_term[1::2])  # odd indices

        self.positional_encoding = tf.convert_to_tensor(self.positional_encoding)

        # positional_encoding.shape = (seq_len, embedding_dims)

    def call(self, input_batch):
        """
        - input_batch: (batch_size, seq_len, embedding_dim)

        - output: (batch_size, seq_len, embedding_dim)
        """
        return input_batch + self.positional_encoding
    
# User defined class for seperate projections of multiple heads

class MultipleHeadAttention(Layer):
    '''
    MultipleHeadAttention Model

    the constructor takes the following:
    - embedding_dims = dimensions of embedding vector
    - num_heads = number of attention heads

    call functions takes following:
    - q: input for query (batch_size, seq_len_q, embedding_dim)
    - k: input for key (batch_size, seq_len_k, embedding_dim)
    - v: input for value (batch_size, seq_len_v, embedding_dim)
    output
    - output: (batch_size, seq_len_q, d_model)
    '''

    def __init__(self, embedding_dim, num_heads):
        super(MultipleHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        assert embedding_dim % self.num_heads == 0, "Embedding dimensions must be divisible by number of heads"

        self.depth = embedding_dim // self.num_heads

        self.wq = [Dense(self.depth) for _ in range(num_heads)]
        self.wk = [Dense(self.depth) for _ in range(num_heads)]
        self.wv = [Dense(self.depth) for _ in range(num_heads)]

        self.dense = Dense(embedding_dim)

    def scaled_dot_product(self, q, k, v, mask = None):
        """
        Calculate the attention weights and apply them to the values.

        - q: (batch_size, seq_len_q, embedding_dim)
        - k: (batch_size, seq_len_k, embedding_dim)
        - v: (batch_size, seq_len_v, embedding_dim)
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch_size, seq_len_q, seq_len_k)


        dk = tf.cast(tf.shape(k)[-1], tf.float32)  # dimensions of embedding dim as float
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None: # for masking values like padding zero
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # normalizing the values

        output = tf.matmul(attention_weights, v)  # (batch_size, seq_len_q, embedding_dim)

        return output, attention_weights

    def generateMask(self, batch_size, seq_len_q, seq_len_k):
        matrix = np.zeros((batch_size, seq_len_q, seq_len_k))

        mask = np.triu(np.ones((seq_len_q, seq_len_k)), k=1)
        matrix[:] = mask

        return tf.convert_to_tensor(matrix)


    def call(self, q, k, v, mask = False):
        '''
        - q: (batch_size, seq_len_q, embedding_dim)
        - k: (batch_size, seq_len_k, embedding_dim)
        - v: (batch_size, seq_len_v, embedding_dim)
        '''
        batch_size = tf.shape(q)[0]

        q = [self.wq[i](q) for i in range(self.num_heads)]  # (num_heads, batch_size, seq_len_q, embedding_dim)
        k = [self.wk[i](k) for i in range(self.num_heads)]
        v = [self.wv[i](v) for i in range(self.num_heads)]


        attention_outputs = []
        for i in range(self.num_heads):
            if mask:
                maskMatrix = self.generateMask(batch_size, q[i].shape[1], k[i].shape[1])
            else:
                maskMatrix = None
            attention_output, _ = self.scaled_dot_product(q[i], k[i], v[i], maskMatrix)
            attention_outputs.append(attention_output)

        # Concatenate the attention outputs
        scaled_attention = tf.concat(attention_outputs, axis=-1)  # (batch_size, seq_len_q, d_model)

        # Final linear layer
        output = self.dense(scaled_attention)  # (batch_size, seq_len_q, d_model)

        return output
    
class LayerNormalization(Layer):
    '''
    Layer normalization model

    Constructor takes the following:
    - embedding_dim: the size of the embedding vector

    Call takes the following:
    - input: the input to the layer normalization

    Call returns the following:
    - output: the normalized input
    '''
    def __init__(self, epsilon=1e-6):
        super(LayerNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        # Learnable gamma and beta parameters for scaling and shifting
        self.gamma = self.add_weight(shape=input_shape[-1:],
                                     initializer='ones',
                                     trainable=True,
                                     name='gamma')
        self.beta = self.add_weight(shape=input_shape[-1:],
                                    initializer='zeros',
                                    trainable=True,
                                    name='beta')

    def call(self, inputs):
        # Compute the mean and variance of the inputs
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=-1, keepdims=True)

        # Normalize the inputs
        normalized_inputs = (inputs - mean) / tf.sqrt(variance + self.epsilon)

        # Scale and shift
        return self.gamma * normalized_inputs + self.beta


class Encoder(Layer):
    '''
    Encoder model

    The constructor takes the following:
    - vocab_size: the size of the vocabulary
    - embedding_dim: the size of the embedding vector
    - num_heads: the number of heads in the multi-head attention
    - num_nodes: the number of nodes in the feed forward network
    - dropout_rate: the dropout rate

    The call method takes the following:
    - inputs_batch: the input to the encoder in form of batch

    The call method returns the following:
    - output: the output of the encoder
    '''
    def __init__(self, vocab_size, embedding_dim, num_heads, num_nodes, dropout_rate = 0.1):
        super(Encoder, self).__init__()

        # Embedding layer
        self.embedding = Embedding(vocab_size, embedding_dim)

        # Positional encoding layer
        self.positionalEncoder = PositionalEncoding(embedding_dim)

        # MultiHeadAttention
        self.multiHeadAttention = MultipleHeadAttention(embedding_dim, num_heads)

        # LayerNormalization
        self.layerNormalization1 = LayerNormalization()
        self.layerNormalization2 = LayerNormalization()

        # Fully Connected Layer
        self.feedForwardNetwork = Sequential([Dense(num_nodes, activation = 'relu'),
                                              Dense(embedding_dim)])

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)


    def call(self, inputs_batch, training=False):
        """
        - inputs_batch: (batch_size, seq_len)

        - embedding: (batch_size, seq_len, embedding_dim)

        - positionalEncoding: (batch_size, seq_len, embedding_dim)

        - attention_output: (batch_size, seq_len, embedding_dim)

        - feed_forward_output: (batch_size, seq_len, embedding_dim)

        - output: (batch_size, seq_len, embedding_dim)
        """

        # Embedding
        embedding = self.embedding(inputs_batch)

        # Positional encoding
        positionalEncoding = self.positionalEncoder(embedding)

        # multi head attention
        attention_output = self.multiHeadAttention(positionalEncoding, positionalEncoding, positionalEncoding)
        attention_output = self.dropout1(attention_output, training = training)
        attention_output += positionalEncoding
        attention_output = self.layerNormalization1(attention_output)

        # feed forward network
        feed_forward_output = self.feedForwardNetwork(attention_output)
        feed_forward_output = self.dropout2(feed_forward_output, training = training)
        feed_forward_output += attention_output
        output = self.layerNormalization2(feed_forward_output)

        return output
    

class Decoder(Layer):
    '''
    Decoder model

    The constructor takes the following:
    - vocab_size: the size of the vocabulary
    - embedding_dim: the size of the embedding vector
    - num_heads: the number of heads in the multi-head attention
    - num_nodes: the number of nodes in the feed forward network
    - dropout_rate: the dropout rate

    The call method takes the following:
    - inputs_batch: the input to the Decoder in form of batch

    The call method returns the following:
    - output: the output of the Decoder
    '''

    def __init__(self, embedding_dim, vocab_size, num_heads, num_nodes, dropout_rate = 0.1):
        super(Decoder, self).__init__()

        self.embedding = Embedding(vocab_size, embedding_dim)

        self.positionalEncoder = PositionalEncoding(embedding_dim)

        self.maskMultiHeadAttention = MultipleHeadAttention(embedding_dim, num_heads)

        self.crossAttention = MultipleHeadAttention(embedding_dim, num_heads)

        # LayerNormalization
        self.layerNormalization1 = LayerNormalization()
        self.layerNormalization2 = LayerNormalization()
        self.layerNormalization3 = LayerNormalization()

        # Fully Connected Layer
        self.feedForwardNetwork = Sequential([Dense(num_nodes, activation = 'relu'),
                                              Dense(embedding_dim)])
        
        # DropOut Layer
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    def call(self, input_batch, encoder_output, training=False):
        """
        - inputs_batch: (batch_size, seq_len)
        - encoder_output: (batch_size, seq_len, embedding_dim)

        - embedding: (batch_size, seq_len, embedding_dim)

        - positionalEncoding: (batch_size, seq_len, embedding_dim)

        - attention_output: (batch_size, seq_len, embedding_dim)

        - feed_forward_output: (batch_size, seq_len, embedding_dim)

        - output: (batch_size, seq_len, embedding_dim)
        """

        # embedding
        embeddings = self.embedding(input_batch)

        # Positional encoding
        positionalEncoding = self.positionalEncoder(embeddings)

        # Mask Multi Head Attention
        maskAttentionOutput = self.maskMultiHeadAttention(positionalEncoding, positionalEncoding, positionalEncoding, mask=True)
        maskAttentionOutput = self.dropout1(maskAttentionOutput, training = training)
        maskAttentionOutput += positionalEncoding
        maskAttentionOutput = self.layerNormalization1(maskAttentionOutput)

        # Cross Attention output
        crossAttentionOutput = self.crossAttention(maskAttentionOutput, encoder_output, encoder_output)
        crossAttentionOutput = self.dropout2(crossAttentionOutput, training = training)
        crossAttentionOutput += maskAttentionOutput
        crossAttentionOutput = self.layerNormalization2(crossAttentionOutput)

        # Feed Forward Network
        feed_forward_ouput = self.feedForwardNetwork(crossAttentionOutput)
        feed_forward_ouput = self.dropout3(feed_forward_ouput, training = training)
        feed_forward_ouput += crossAttentionOutput
        output = self.layerNormalization3(feed_forward_ouput)

        return output
    

class Transformer(Model):
    '''
    Transformer model

    Constructor takes the following:
    - vocab_size_input: the size of the vocabulary (inputs)
    - vocab_size_output: the size of the vocabulary (outputs)
    - embedding_dim: the size of the embedding vector
    - num_heads_encoder: the number of attention heads in encoder
    - num_heads_decoder: the number of attention heads in decoder
    - num_encoder_layers: the number of encoder layers
    - num_decoder_layers: the number of decoder layers
    - num_nodes_encoder: the number of nodes in the feed-forward network of encoder
    - num_nodes_decoder: the number of nodes in the feed-forward network of decoder
    - dropout_rate: the dropout rate

    Call takes the following:
    - encoder_input: input to the encoder
    - decoder_input: input to the decoder
    - training: boolean indicating training model

    Call returns the following:
    - output: the output of the transformer model
    '''

    def __init__(self, vocab_size_input, vocab_size_output, embedding_dim, num_heads_encoder, num_heads_decoder, num_encoder_layers, num_decoder_layers, num_nodes_encoder, num_nodes_decoder, dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder_layers = [
            Encoder(vocab_size_input, embedding_dim, num_heads_encoder, num_nodes_encoder, dropout_rate)
            for _ in range(num_encoder_layers)
        ]

        self.decoder_layers = [
            Decoder(embedding_dim, vocab_size_output, num_heads_decoder, num_nodes_decoder, dropout_rate)
            for _ in range(num_decoder_layers)
        ]

        self.final_layer = Dense(vocab_size_output)  # Final linear layer to project decoder output to vocab size

    def call(self, encoder_input, decoder_input, training=False):
        """
        - encoder_input: (batch_size, seq_len)
        - decoder_input: (batch_size, seq_len)
        - output: (batch_size, seq_len, vocab_size)
        """

        # Pass through each encoder layer
        encoder_output = encoder_input
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, training = training)

        # Pass through each decoder layer
        decoder_output = decoder_input
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(decoder_output, encoder_output, training=training)

        # Pass through the final linear layer to get the logits for each token in the sequence
        output = self.final_layer(decoder_output)

        return output
