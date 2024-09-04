import tensorflow as tf
from tensorflow import keras

class CreateMask:
    
    def create_padding_mask(self, seq):
        """
        Creates a padding mask for a given sequence.

        Args:
            seq (tensor): A tensor of shape (batch_size, seq_len) containing the sequence.

        Returns:
            A tensor of shape (batch_size, 1, seq_len) containing a mask that is 1 where the sequence is padded, and 0 otherwise.
        """
        # Convert the sequence to a boolean tensor where True indicates a pad token (value 0).
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # Add an extra dimension to the mask to add the padding to the attention logits.
        return seq[:, tf.newaxis, :]
    
    def create_look_ahead_mask(self, size):
        """
        Creates a look-ahead mask used during training the decoder of a transformer.

        Args:
            size (int): The size of the mask.

        Returns:
            tf.Tensor: A lower triangular matrix of shape (size, size) with ones on the diagonal
                and zeros below the diagonal.
        """
        # create a matrix with ones on the diagonal and zeros below the diagonal
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

        return mask
    

    def create_masks(self, inputs, targets):
        """
        Creates masks for the input sequence and target sequence.

        Args:
            inputs: Input sequence tensor.
            targets: Target sequence tensor.

        Returns:
            A tuple of three masks: the encoder padding mask, the combined mask used in the first attention block, 
            and the decoder padding mask used in the second attention block.
        """

        # Create the encoder padding mask.
        enc_padding_mask = self.create_padding_mask(inputs)

        # Create the decoder padding mask.
        # this passed to decoder crossAttentionLayer so it should be calculated on encoder_output (same as encoder_input)
        dec_padding_mask = self.create_padding_mask(inputs)
        
        dec_target_padding_mask = self.create_padding_mask(targets)
        
        # Create the look ahead mask for the first attention block.
        # It is used to pad and mask future tokens in the tokens received by the decoder.
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(targets)[1])


        # Combine the look ahead mask and decoder target padding mask for the first attention block.
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask





class Metrices:
    
    def __init__(self):
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    
    def loss_function(self, true_value, predicted):
        
        """
        Calculate the loss value for a given target sequence.

        Args:
            true_values: The true target sequence.
            predictions: The predicted target sequence.

        Returns:
            float: The loss value for the given target sequence.
        """
        # Create a mask to exclude the padding tokens
        mask = tf.math.logical_not(tf.math.equal(true_value, 0))

        # Compute the loss value using the loss object
        loss_ = self.loss_object(true_value, predicted)

        # Apply the mask to exclude the padding tokens (makes padding loss zero)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        # Calculate the mean loss value (total_loss / number_of_losses)
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    
    def accuracy_function(true_value, predicted):
        """
        Calculate the accuracy for a given target sequence.

        Args:
            true_values (tf.Tensor): The true target sequence.
            predictions (tf.Tensor): The predicted target sequence.

        Returns:
            float: The accuracy value for the given target sequence.
        """
        # Compute the accuracies using the true and predicted target sequences
        accuracies = tf.equal(true_value, tf.argmax(predicted, axis=-1))

        # Create a mask to exclude the padding tokens
        mask = tf.math.logical_not(tf.math.equal(true_value, 0))

        # Apply the mask to exclude the padding tokens from the accuracies
        accuracies = tf.math.logical_and(mask, accuracies)
        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)

        # Calculate the mean accuracy value
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)
    



class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """
    A custom learning rate schedule that uses a combination of
    a square root inverse decay and a warmup schedule.

    Args:
        embedding_dim (int): The dimension of the embedding.
        warmup_steps (int): The number of steps used for warmup.

    Returns:
        float: The learning rate value at a given step.
    """

    def __init__(self, embedding_dim, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.embedding_dim = tf.cast(embedding_dim, dtype=tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, dtype=tf.float32)

    def __call__(self, step):
        """
        Compute the learning rate value for a given step using
        a combination of square root inverse decay and warmup.

        Args:
            step (int): The current step number.

        Returns:
            float: The learning rate value at the current step.
        """
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.embedding_dim) * tf.math.minimum(arg1, arg2)





class Train:
    
    def __init__(self, batch_size, max_input_len, max_output_len, transformer,
                   optimizer = keras.optimizers.Adam):
        
        # training signature list containing specifications (shape, dtype) of all the inputs sequences passed to training
        self.train_signature = [
            tf.TensorSpec((batch_size, max_input_len), dtype = tf.float32),
            tf.TensorSpec((batch_size, max_output_len), dtype = tf.float32)
        ]

        # Define the training metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        
        # Store the optimizer class
        self.optimizer_class = optimizer
        
        # Initialize the optimizer with default parameters
        self.optimizer = self.optimizer_class()
        
        self.transformer = transformer
        
        # Wrap the train_step function with tf.function and input signature
        self.train_step = tf.function(self.train_step, input_signature=self.train_signature)

        
    def set_optimizer(self, learning_rate=0.01, beta_1=0.9, beta_2=0.98, epsilon=1e-9):
        """
        Set a custom optimizer with specific parameters.

        Args:
            learning_rate (float): The learning rate for the optimizer.
            beta_1 (float): The exponential decay rate for the first moment estimates.
            beta_2 (float): The exponential decay rate for the second-moment estimates.
            epsilon (float): A small constant for numerical stability.
        """
        self.optimizer = self.optimizer_class(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
        
        
#     @tf.function(input_signature=self.train_signature)
    def train_step(self, input_seq, output_seq):
        """
        Function to perform a single training step.

        Args:
        input_seq (tf.Tensor): The input tensor for the encoder.
        output_seq (tf.Tensor): The target tensor for the decoder.

        Returns:
        None.
        """

        # remove last word from each sequence of input (since we need to predict this word at last and no further prediction is done)
        decoder_input = output_seq[:, :-1]


        # remove first word from each sequence of expected output (since we don't predict the first word)
        expected_output = output_seq[:, 1:]

        mask = CreateMask()
        enc_padding_mask, combined_mask, dec_padding_mask = mask.create_masks(input_seq, decoder_input)

        evaluate = Metrices()

        with tf.GradientTape() as tape:

            prediction = self.transformer(input_seq,
                                     decoder_input,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask,
                                     training=True)

            loss = evaluate.loss_function(expected_output, prediction)


        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))


        # Update the training loss and accuracy metrics
        self.train_loss(loss)
        self.train_accuracy(expected_output, prediction)