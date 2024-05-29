# Import libraries
# ------------------------------------------------------------------------------
import tensorflow as tf


# Predictor
# ------------------------------------------------------------------------------
class Predictor(tf.Module):
  def __init__(self, transformer):
    self.transformer = transformer

  def __call__(self, trace, mapping, max_length=100):
    # The input sentence - adding the `[START]` and `[END]` tokens.
    assert isinstance(trace, tf.Tensor)
    if len(trace.shape) == 0:
      trace = trace[tf.newaxis]

    encoder_input = trace

    # Initlize output with the start token
    start = tf.cast(tf.constant(mapping['<start>'])[tf.newaxis], tf.int64)
    end = tf.cast(tf.constant(mapping['<end>'])[tf.newaxis], tf.int64)

    # Create output arrays
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, start)

    for i in tf.range(max_length-1): # Because the start token is inserted above
      output = tf.transpose(output_array.stack())
      predictions = self.transformer([encoder_input, output], training=False)

      # Select the last token from the `seq_len` dimension.
      predictions = predictions[:, -1:, :]  # Shape `(variables['batch_size'], 1, variables['vocab_size'])`.

      predicted_id = tf.argmax(predictions, axis=-1)

      # Concatenate the `predicted_id` to the output which is given to the decoder as its input.
      output_array = output_array.write(i+1, predicted_id[0])

      if predicted_id == end:
        break

    output = tf.transpose(output_array.stack())
    # The output shape is `(1, tokens)`.

    return output