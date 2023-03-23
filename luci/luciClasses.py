class LuciPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, seq_length=2048):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = self._positional_encoding(seq_length)

    def _positional_encoding(self, length):
        depth = self.d_model // 2
        positions = tf.range(length)[:, tf.newaxis] # (seq_length, 1)
        angles = tf.range(depth)[tf.newaxis, :] / tf.cast(depth, tf.float32) # (1, d_model/2)
        angle_rads = positions * angles * (1 / tf.pow(10000, 2 * angles)) # (seq_length, d_model/2)
        sin_cos = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1) # (seq_length, d_model)
        return tf.expand_dims(tf.cast(sin_cos, tf.float32), axis=0)

    def call(self, x):
        x = self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[:, :tf.shape(x)[1], :]
        return x
