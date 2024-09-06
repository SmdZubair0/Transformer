import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Dense, Embedding, Dropout
from keras.models import Sequential, Model

class PositionalEncoding(Layer):
    '''
    Positional encoding model

    Constructor takes the following:
    
    args
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


        self.positional_encoding[:, 0::2] = np.sin(pos / div_term[0::2])  # even indices
        self.positional_encoding[:, 1::2] = np.cos(pos / div_term[1::2])  # odd indices

        self.positional_encoding = tf.convert_to_tensor(self.positional_encoding)
        self.positional_encoding = tf.cast(self.positional_encoding, tf.float32)

        # positional_encoding.shape = (seq_len, embedding_dims)

    def call(self, input_batch):
        """
        args
            - input_batch: (batch_size, seq_len, embedding_dim)
            
        - output: (batch_size, seq_len, embedding_dim)
        """
        
        seq_len = tf.shape(input_batch)[1]
        pos = self.positional_encoding[:seq_len, :]
        
        return input_batch + tf.expand_dims(pos, axis = 0)


class MultipleHeadAttention(Layer):
    '''
    MultipleHeadAttention Model

    the constructor takes the following:
    args
        - embedding_dims = dimensions of embedding vector
        - num_heads = number of attention heads

    call functions takes following:
    args
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
        
        args
            - q: query vector (batch_size, seq_len_q, depth)
            - k: key vector (batch_size, seq_len_k, depth)
            - v: value vector (batch_size, seq_len_v, depth)
            (optional)
            - mask: (batch_size, 1, seq_len_q) or (batch_size, seq_len_q, seq_len_q)
            
        - output: attention value (batch_size, seq_len_q, depth)
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch_size, seq_len_q, seq_len_k)



        dk = tf.cast(tf.shape(k)[-1], tf.float32)  # dimensions of embedding dim as float
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None: # for masking values like padding zero
            scaled_attention_logits += (mask * -1e9)
            
            
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # normalizing the values (batch_size, seq_len_q, seq_len_k)
        output = tf.matmul(attention_weights, v)  # (batch_size, seq_len_q, depth)
        
        return output, attention_weights
    

    def call(self, q, k, v, mask = None):
        '''
        args
            - q: (batch_size, seq_len_q, embedding_dim)
            - k: (batch_size, seq_len_k, embedding_dim)
            - v: (batch_size, seq_len_v, embedding_dim)
        '''
        batch_size = tf.shape(q)[0]

        q = [self.wq[i](q) for i in range(self.num_heads)]  # (num_heads, batch_size, seq_len_q, depth)
        k = [self.wk[i](k) for i in range(self.num_heads)]
        v = [self.wv[i](v) for i in range(self.num_heads)]


        attention_outputs = []
        
        for i in range(self.num_heads):
                
            attention_output, _ = self.scaled_dot_product(q[i], k[i], v[i], mask)
            attention_outputs.append(attention_output)

        # Concatenate the attention outputs
        scaled_attention = tf.concat(attention_outputs, axis=-1)  # (batch_size, seq_len_q, embedding_dim)

        # Final linear layer
        output = self.dense(scaled_attention)  # (batch_size, seq_len_q, embedding_dim)

        return output
    
class LayerNormalization(Layer):
    '''
    Layer normalization model

    Call takes the following:
    - input: the input to the layer normalization (batch_size, seq_len, embedding_dim)

    Call returns the following:
    - output: the normalized input
    '''
    def __init__(self, epsilon=1e-6):
        super(LayerNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        '''
        - input_shape: (batch_size, seq_len, embedding_dim)
        '''
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
        '''
        args
            - inputs: (batch_size, seq_len, embedding_dim)
        
        - output: (batch_size, seq_len, embedding_dim)
        '''
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
    
    args
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


    def call(self, input_batch, mask=None, training=False):
        """
        args
            - input_batch: input for encoder (batch_size, seq_len, embedding_dim)
            - mask: padding_mask for encoder (batch_size, 1, seq_len)

        - output: (batch_size, seq_len, embedding_dim)
        """

        positionalEncoding = input_batch

        # multi head attention
        attention_output = self.multiHeadAttention(
            q = positionalEncoding,
            k = positionalEncoding,
            v = positionalEncoding,
            mask = mask
        )
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
    args
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

    def __init__(self, vocab_size, embedding_dim, num_heads, num_nodes, dropout_rate = 0.1):
        super(Decoder, self).__init__()


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

    def call(self, input_batch, encoder_output, look_ahead_mask, padding_mask, training=False):
        """
        args
            - inputs_batch: input for decoder (batch_size, seq_len_dec, embedding_dim)
            - encoder_output: output generated by encoder (batch_size, seq_len_en, embedding_dim)
            - look_ahead_mask: mask for masked attention (batch_size, seq_len_dec, seq_len_dec)
            - padding_mask: mask for padding (batch_size, 1, seq_len_dec)
            - training: True while training
        
        - output: 
        """

        positionalEncoding = input_batch

        # Mask Multi Head Attention
        maskAttentionOutput = self.maskMultiHeadAttention(
            q = positionalEncoding,
            k = positionalEncoding,
            v = positionalEncoding,
            mask = look_ahead_mask
        )
        maskAttentionOutput = self.dropout1(maskAttentionOutput, training = training)
        maskAttentionOutput += positionalEncoding
        maskAttentionOutput = self.layerNormalization1(maskAttentionOutput)

        # Cross Attention output
        crossAttentionOutput = self.crossAttention(
            q = maskAttentionOutput,
            k = encoder_output,
            v = encoder_output,
            mask = padding_mask
        )
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
    
    args
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
    
    args:
        - encoder_input: input to the encoder
        - decoder_input: input to the decoder
        - training: boolean indicating training model


    Call returns the following:
    
    - output: the output of the transformer model
    '''

    def __init__(self, vocab_size_input, vocab_size_output, embedding_dim, num_heads_encoder, num_heads_decoder, num_encoder_layers, num_decoder_layers, num_nodes_encoder, num_nodes_decoder, dropout_rate=0.1):
        super(Transformer, self).__init__()
        
        self.embedding_encoder = Embedding(vocab_size_input, embedding_dim)
        self.embedding_decoder = Embedding(vocab_size_output, embedding_dim)
        
        self.positional_encoding = PositionalEncoding(embedding_dim)

        self.encoder_layers = [
            Encoder(
                vocab_size = vocab_size_input,
                embedding_dim = embedding_dim,
                num_heads = num_heads_encoder,
                num_nodes = num_nodes_encoder,
                dropout_rate = dropout_rate
            )
            for _ in range(num_encoder_layers)
        ]

        self.decoder_layers = [
            Decoder(
                vocab_size = vocab_size_output,
                embedding_dim = embedding_dim,
                num_heads = num_heads_decoder,
                num_nodes = num_nodes_decoder,
                dropout_rate = dropout_rate
            )
            for _ in range(num_decoder_layers)
        ]

        self.final_layer = Dense(vocab_size_output)  # Final linear layer to project decoder output to vocab size
        


    def call(self, encoder_input, decoder_input, enc_padding_mask, combined_mask, dec_padding_mask, training=False):
        """
        args
            - encoder_input: (batch_size, in_seq_len)
            - decoder_input: (batch_size, out_seq_len)
            - enc_padding_mask: (batch_size, 1, in_seq_len)
            - combined_mask: (batch_size, out_seq_len, out_seq_len)
            - dec_padding_mask: (batch_size, 1, out_seq_len)
            - Training: True while training
            
        - output: (batch_size, seq_len, vocab_size)
        """
        
        encoder_embedding = self.embedding_encoder(encoder_input)  # (batch_size, in_seq_len, embedding_dim)
        encoder_positional_embedding = self.positional_encoding(encoder_embedding)  # (batch_size, in_seq_len, embedding_dim)
        
        # Pass through each encoder layer
        encoder_output = encoder_positional_embedding
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, enc_padding_mask, training = training)
            
        
        decoder_embedding = self.embedding_decoder(decoder_input)  # (batch_size, out_seq_len, embedding_dim)
        decoder_positional_embedding = self.positional_encoding(decoder_embedding)  # (batch_size, out_seq_len, embedding_dim)
        
        # Pass through each decoder layer
        decoder_output = decoder_positional_embedding
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(decoder_output, encoder_output, combined_mask, dec_padding_mask, training=training)
            
            
        # Pass through the final linear layer to get the logits for each token in the sequence
        output = self.final_layer(decoder_output)

        return output
