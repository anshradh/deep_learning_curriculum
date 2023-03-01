- Here are some first principle questions to answer:
    - What is different architecturally from the Transformer, vs a normal RNN, like an LSTM? (Specifically, how are recurrence and time managed?)
      Time is managed throughout the sequence dimension of the input - there's no recurrence, since the hidden state at each position can attend to every previous hidden state (as well as itself).
    - Attention is defined as, Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V. What are the dimensions for Q, K, and V? Why do we use this setup? What other combinations could we do with (Q,K) that also output weights?
      Q, K, and V are all the same size and are (seq_len, d_head) shape for each head. We use this setup because it allows us to represent how much each token wants to attent to every previous token (and itself) and what happens if a given token is attended to. We could also calculate cosine similarities between Q and K, or do a dot product without the scaling (the scaling maintains unit variance of activations), or introduce a weight matrix between Q and K.
    - Are the dense layers different at each multi-head attention block? Why or why not?
      Yes, they are different. This allows the model to learn different representations for each head at each block.
    - Why do we have so many skip connections, especially connecting the input of an attention function to the output? Intuitively, what if we didn't?
      The skip connections allow the network to learn even if it's very deep, since the loss gradient can flow backward through the entire network without vanishing or exploding.
