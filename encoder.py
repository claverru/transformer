import keras
from keras_multi_head import MultiHeadAttention
from keras_layer_normalization import LayerNormalization
from keras_position_wise_feed_forward import FeedForward
from keras_pos_embd import TrigPosEmbedding


"""pip install keras-transformer
Thanks to CyberZHG
"""

def encoder(seq_len, m_features, d_model, n_heads, dff, rate=0.1, encoder=None):
	"""Basic Attention Encoder. It can be concatenated with a previous encoder by passing it as argument."""
	if encoder == None:
		in_seq = keras.layers.Input(shape=(seq_len, m_features))
		in_seq = LayerNormalization()(in_seq)
	else::
		in_seq = encoder.output
	linear = keras.layers.Dense(units=d_model)(norm_0)
	pos = TrigPosEmbedding(mode=TrigPosEmbedding.MODE_ADD)(linear)
	mha = MultiHeadAttention(head_num=n_heads)(pos)
	mha_drop = keras.layers.Dropout(rate=rate)(mha)
	add_1  = keras.layers.Add()([pos, mha_drop])
	norm_1 = LayerNormalization()(add_1)
	ff = FeedForward(dff)(norm_1)
	ff_drop = keras.layers.Dropout(rate=rate)(ff)
	add_2 = keras.layers.Add()([ff_drop, norm_1])
	out = LayerNormalization()(add_2)
	return keras.Model(in_seq, out) if encoder == None else keras.Model(encoder.input, out)
