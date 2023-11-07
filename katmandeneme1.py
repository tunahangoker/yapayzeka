import numpy as np 
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from numpy import random

maxlen = 20
vocab_size = 100
number_of_categories = 20
rawdata = random.randint(vocab_size, size=(200, maxlen+1))
X = rawdata[:,:-1]
c = random.randint(number_of_categories, size=(200,1))
y = rawdata[:,1:]
     

train_ds= tf.data.Dataset.from_tensor_slices((X,c,y))
train_ds.element_spec #buda özelliklerini gösterir floatmı vb
     
def re_struct(X,c,y):
  return (X,c),y
train_ds = train_ds.map(re_struct) #değişkenlerin türünü gösterir
##########################################################################################################

#optime etme baya fark eder 
batch_size=64
train_ds=train_ds.batch(batch_size=batch_size,drop_remainder=True, 
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_ds=train_ds.cache()
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
     
###########################################################################################################
train_ds.element_spec
     

def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = tf.range(n_dest)[:, None] #  
    j = tf.range(n_src)  #n_src ye kadar olan sayıları 0 dan başlayarak sıralar 1 2 3 4 5 vb 
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)   #değişkenin türünü değiştirir   (1.si hangi değişken 2. si hangi türe çevrilsin int vb)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(                                      #dizileri birleştirir alt alta yazabilirsin istediklerini birleştir             
                                                           #tf.concat(5,9,11)  alt satırada farklı versiyonu bunları tek bir dizi yapar  boyutu aynı kalır sadece birleştirir                   
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0   #tf.expand_dims  dizinin boyutunu 1 arttırır hepsini bir diziye toplayıp oran hesabı vb
    )
    return tf.tile(mask, mult)

#ikinci katman 
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__()
        self.embed_dim=embed_dim
        self.num_heads = num_heads
        self.ff_dim =ff_dim
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)    
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)
    


#birinci katman
class TokenPositionAndCategoricalEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, number_of_categories, embed_dim, **kwargs):
        super(TokenPositionAndCategoricalEmbedding, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.number_of_categories = number_of_categories
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.category_emb = layers.Embedding(input_dim=number_of_categories, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, X, c):          #istediğimiz işlemi burda yapıyoruz
        maxlen = tf.shape(X)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        X = self.token_emb(X)
        c= self.category_emb(c)
        return X + positions + c
    

embed_dim = 256  # Embedding size for each token
num_heads = 2  # Number of attention heads
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer



# bu yazdığımzı layerları  burda oluşturduk layer in layer yaptık 
def create_model():
    inputs_tokens = layers.Input(shape=(maxlen,), dtype=tf.int32) #maxlen adamın datalarının olduğu yer datalarını burdan vericen 
    inputs_category = layers.Input(shape=(1,), dtype=tf.int32, name="inputs_category")
    # the first custom layer for embedding 
    embedding_layer = TokenPositionAndCategoricalEmbedding(maxlen, vocab_size, 
                                                           number_of_categories, 
                                                           embed_dim)
    x = embedding_layer(inputs_tokens,inputs_category)
    # the second custom layer for GPT-kind-of transformer decoder block
    transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
    x = transformer_block(x)
    outputs = layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=[inputs_tokens,inputs_category], outputs=[outputs, x])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        "adam", loss=[loss_fn, None],
    )  # No loss and optimization based on word embeddings from transformer block
    return model
my_model=create_model()
my_model.summary()