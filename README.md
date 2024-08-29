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

'''
    Layer normalization model

    Constructor takes the following:
    - embedding_dim: the size of the embedding vector

    Call takes the following:
    - input: the input to the layer normalization

    Call returns the following:
    - output: the normalized input
'''