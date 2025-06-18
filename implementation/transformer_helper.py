import numpy as np # type: ignore

class TransformerHelper:
    
    @staticmethod
    def compute_loss_and_gradient(model, source_sequence, target_sequence, target_token_seq_idx, target_token_vocab_idx):
        # Split the full target sequence into tokens
        target_tokens = target_sequence.split()

        # Create the decoder input by taking all tokens before the current target token,
        # and prepending <start> at the beginning
        decoder_input_seq = ' '.join(['<start>'] + target_tokens[:target_token_seq_idx])

        # Run a full forward pass through the model using the current source and partial target input
        output_probabilities = TransformerHelper.full_forward_pass(model, 
                                                                   source_sequence, 
                                                                   decoder_input_seq, 
                                                                   store_intermediate_results=True)

        # Extract the probability assigned to the correct target token
        probability_of_target_token = output_probabilities[-1, target_token_vocab_idx]

        # Compute the negative log likelihood loss for this token (with epsilon for stability)
        loss = -np.log(probability_of_target_token + 1e-9)

        # Compute the gradient of the loss with respect to all model parameters
        gradients = TransformerHelper.compute_the_gradient(model, output_probabilities, target_token_vocab_idx)

        return loss, gradients



    @staticmethod
    def full_forward_pass(model, 
                          source_sequence, 
                          decoder_input_seq, 
                          store_intermediate_results=False):

        encoder_layer = model.encoder_layer
        decoder_layer = model.decoder_layer

        # ===============================
        #           ENCODER
        # ===============================

        # Embed and positionally encode the input tokens
        encoder_input = TransformerHelper.positionally_encoded_embeddings(
            encoder_layer, 
            source_sequence, 
            store_intermediate_results
        )

        num_att_heads = 3  # Number of attention heads for all multi-head attention blocks

        # Run multi-head self-attention over the encoder input
        encoder_self_att_output = TransformerHelper.multihead_attention(
            encoder_layer,
            encoder_input,
            num_att_heads, 
            store_intermediate_results,
            cross_att_context=None,   # Self-attention uses input as its own context
            attn_type='self'
        )

        # Add residual connection and normalize
        encoder_norm_self_att = TransformerHelper.layer_norm(
            encoder_layer,
            encoder_input + encoder_self_att_output, 
            store_intermediate_results,
            part='self_att'
        )

        # Run the position-wise feed-forward network
        encoder_ffn_layer_2 = TransformerHelper.feed_forward(
            encoder_layer,
            encoder_norm_self_att, 
            store_intermediate_results
        )

        # Add residual connection and normalize again
        encoder_norm_ffn = TransformerHelper.layer_norm(
            encoder_layer,
            encoder_ffn_layer_2 + encoder_norm_self_att,
            store_intermediate_results,
            part='ffn'
        )

        # Final encoder output
        encoder_output = encoder_norm_ffn

        # ===============================
        #           DECODER
        # ===============================

        # Embed and positionally encode the decoder input tokens
        decoder_input = TransformerHelper.positionally_encoded_embeddings(
            decoder_layer,
            decoder_input_seq,
            store_intermediate_results
        )

        # Masked self-attention (prevents tokens from attending to future positions)
        decoder_masked_att_output = TransformerHelper.multihead_attention(
            decoder_layer,
            decoder_input,
            num_att_heads,
            store_intermediate_results,
            cross_att_context=None,   # No external context, it's self-attention
            attn_type='masked'
        )

        # Add residual and normalize
        decoder_norm_masked_att = TransformerHelper.layer_norm(
            decoder_layer,
            decoder_input + decoder_masked_att_output,
            store_intermediate_results,
            part='masked_att'
        )

        # Cross-attention: decoder attends to encoder outputs
        decoder_cross_att_output = TransformerHelper.multihead_attention(
            decoder_layer,
            decoder_norm_masked_att,
            num_att_heads,
            store_intermediate_results,
            cross_att_context=encoder_output,
            attn_type='cross'
        )

        # Add residual and normalize
        decoder_norm_cross_att = TransformerHelper.layer_norm(
            decoder_layer,
            decoder_norm_masked_att + decoder_cross_att_output,
            store_intermediate_results,
            part='cross_att'
        )

        # Feed-forward in decoder
        decoder_ffn_layer_2 = TransformerHelper.feed_forward(
            decoder_layer,
            decoder_norm_cross_att,
            store_intermediate_results
        )

        # Final residual connection and normalization
        decoder_norm_ffn = TransformerHelper.layer_norm(
            decoder_layer,
            decoder_ffn_layer_2 + decoder_norm_cross_att,
            store_intermediate_results,
            part='ffn'
        )

        # Final decoder output
        decoder_output = decoder_norm_ffn

        # ===============================
        #      OUTPUT PROJECTION
        # ===============================

        # Project decoder output to vocabulary space using the transpose of the embedding matrix
        logits = decoder_output @ decoder_layer.embedding_matrix.T  # shape (tgt_len, vocab_size)

        # Softmax to get probability distribution over vocabulary
        output_probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

        return output_probabilities



    @staticmethod
    def positionally_encoded_embeddings(module, sequence, store_intermediate_results, max_len=100):
        # Tokenize the input string
        if not isinstance(sequence, str):
            raise ValueError("Input 'sequence' must be a string")
        tokens = sequence.strip().split()  # Simple whitespace tokenization

        # Get the correct vocabulary (input or output) based on the module type
        if hasattr(module, 'input_vocab'):
            token_to_idx = module.input_vocab
        elif hasattr(module, 'output_vocab'):
            token_to_idx = module.output_vocab
        else:
            raise ValueError(f"Module of type {type(module)} has no input or output vocab")

        # Convert tokens into vocabulary indices, raising an error if any token is unknown
        indices = []
        for token in tokens:
            if token not in token_to_idx:
                raise ValueError(f"Token '{token}' not found in vocab: {list(token_to_idx.keys())}")
            indices.append(token_to_idx[token])
        indices = np.array(indices, dtype=int)

        # Look up embeddings using the embedding matrix
        embedding_matrix = module.embedding_matrix              # shape: (vocab_size, d_model)
        embeddings = embedding_matrix[indices]                  # shape: (seq_len, d_model)

        # Generate sinusoidal positional encodings and add them to token embeddings
        seq_len, d_model = embeddings.shape
        position_enc = TransformerHelper.sinusoidal_positional_encoding(seq_len, d_model)
        embeddings_pe = embeddings + position_enc               # shape: (seq_len, d_model)

        # Optionally store the result for later access (e.g., during backprop)
        if store_intermediate_results:
            module.position_encoded = embeddings_pe

        return embeddings_pe



    @staticmethod
    def sinusoidal_positional_encoding(seq_len, d_model):
        # Create a matrix of positions: shape (seq_len, 1)
        position = np.arange(seq_len)[:, np.newaxis]

        # Compute the divisor term for each dimension (for sine/cosine frequencies)
        # Shape: (d_model/2,)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        # Initialize the full positional encoding matrix with zeros
        pe = np.zeros((seq_len, d_model))

        # Apply sine to even dimensions (0, 2, 4, ...)
        pe[:, 0::2] = np.sin(position * div_term)

        # Apply cosine to odd dimensions (1, 3, 5, ...)
        pe[:, 1::2] = np.cos(position * div_term)

        return pe
        


    @staticmethod
    def multihead_attention(
            module,
            input_tensor,                 # (seq_len, d_model)
            num_att_heads,
            store_intermediate_results,
            cross_att_context=None,       # None for self/decoder-masked, encoder_output for cross
            attn_type='self'              # 'self', 'masked', or 'cross'
        ):
        outputs = []                     # Store output from each attention head
        all_Q, all_K, all_V, all_scores, all_softmax = [], [], [], [], []

        # Use the provided context if cross-attention, otherwise attend to self
        context = cross_att_context if cross_att_context is not None else input_tensor

        # Process each attention head independently
        for h in range(1, num_att_heads + 1):
            # Retrieve weight matrices for this head
            W_Q = getattr(module, f'W_queries_{attn_type}_att_h_{h}')
            W_K = getattr(module, f'W_keys_{attn_type}_att_h_{h}')
            W_V = getattr(module, f'W_values_{attn_type}_att_h_{h}')

            # Compute query, key, and value matrices
            Q = input_tensor @ W_Q              # shape: (seq_len, head_dim)
            K = context @ W_K                   # shape: (context_len, head_dim)
            V = context @ W_V                   # shape: (context_len, head_dim)
            d_k = Q.shape[-1]                   # head_dim

            # Compute scaled dot-product attention scores
            scores = Q @ K.T / np.sqrt(d_k)     # shape: (seq_len, context_len)

            # Apply causal mask in decoder self-attention to prevent looking ahead
            if attn_type == 'masked':
                mask = np.triu(np.ones_like(scores), k=1).astype(bool)
                scores[mask] = -1e9             # large negative value to zero out after softmax

            # Apply numerically stable softmax: subtract max before exp
            scores_stable = scores - np.max(scores, axis=-1, keepdims=True)
            softmax_scores = np.exp(scores_stable)
            softmax_scores = softmax_scores / np.sum(softmax_scores, axis=-1, keepdims=True)

            # Compute weighted sum of values using attention weights
            out = softmax_scores @ V            # shape: (seq_len, head_dim)

            # Optionally store all intermediate results for backpropagation
            if store_intermediate_results:
                setattr(module, f'queries_{attn_type}_att_h_{h}', Q)
                setattr(module, f'keys_{attn_type}_att_h_{h}', K)
                setattr(module, f'values_{attn_type}_att_h_{h}', V)
                setattr(module, f'scores_{attn_type}_att_h_{h}', scores)
                if attn_type == 'masked':
                    setattr(module, f'masked_scores_{attn_type}_att_h_{h}', scores)
                setattr(module, f'softmax_{attn_type}_att_h_{h}', softmax_scores)
                setattr(module, f'output_{attn_type}_att_h_{h}', out)

            # Save this head's output
            outputs.append(out)
            all_Q.append(Q)
            all_K.append(K)
            all_V.append(V)
            all_scores.append(scores)
            all_softmax.append(softmax_scores)

        # Concatenate outputs from all heads along the feature dimension
        multihead_out = np.concatenate(outputs, axis=-1)  # shape: (seq_len, d_model)

        # Apply a final linear projection to mix head outputs
        W_output = getattr(module, f'W_output_{attn_type}_att')
        output = multihead_out @ W_output                 # shape: (seq_len, d_model)

        # Optionally store the final multi-head attention output
        if store_intermediate_results:
            setattr(module, f'{attn_type}_att_output', output)

        return output



    @staticmethod
    def layer_norm(
            module,
            input_tensor,                # (seq_len, d_model)
            store_intermediate_results,  # bool
            part                         # e.g., 'self_att', 'masked_att', 'cross_att', 'ffn'
        ):

        # Select the learnable scale (gamma) and shift (beta) parameters for this part
        gamma = getattr(module, f'gamma_norm_{part}')
        beta  = getattr(module, f'beta_norm_{part}')

        # Compute mean and standard deviation for each token embedding (along feature dimension)
        mean = np.mean(input_tensor, axis=-1, keepdims=True)             # shape: (seq_len, 1)
        std  = np.std(input_tensor, axis=-1, keepdims=True) + 1e-6       # epsilon added to prevent divide-by-zero

        # Normalize each feature vector to have mean 0 and std 1
        normalized = (input_tensor - mean) / std

        # Apply scale and shift: output = gamma * normalized + beta
        output = gamma * normalized + beta                               # shape: (seq_len, d_model)

        # Optionally store intermediate values for use in backpropagation
        if store_intermediate_results:
            setattr(module, f'added_{part}', input_tensor)               # residual input before norm
            setattr(module, f'norm_{part}', output)                      # output after norm and affine transform

        return output



    @staticmethod
    def feed_forward(
            module,
            input_tensor,                # (seq_len, d_model)
            store_intermediate_results   # bool
        ):
        # First linear transformation (affine layer)
        W1 = module.W_ffn_layer_1            # shape: (d_model, d_model)
        layer1 = input_tensor @ W1           # shape: (seq_len, d_model)

        # Apply ReLU activation: replaces negative values with 0
        relu = np.maximum(0, layer1)         # shape: (seq_len, d_model)

        # Second linear transformation
        W2 = module.W_ffn_layer_2            # shape: (d_model, d_model)
        layer2 = relu @ W2                   # shape: (seq_len, d_model)

        # Optionally store intermediate values for backpropagation
        if store_intermediate_results:
            module.ffn_layer_1 = layer1      # Output before activation
            module.relu_ffn = relu           # Activated values
            module.ffn_layer_2 = layer2      # Final feed-forward output

        return layer2



    @staticmethod
    def compute_the_gradient(model, output_probabilities, target_token_vocab_idx):
        grads = {}
        decoder = model.decoder_layer
        encoder = model.encoder_layer

        # ---------------------------------------
        # 1. Gradient w.r.t. final softmax and output projection
        # ---------------------------------------

        # Set gradients for all but last timestep to zero
        output_probabilities[:-1, :] = 0

        # Subtract 1 from the predicted probability at the target token index
        # This turns the softmax output into ∂L/∂logits using cross-entropy gradient
        output_probabilities[-1, target_token_vocab_idx] -= 1

        # This is now the gradient of the loss w.r.t. the logits
        loss_wrt_vocabulary_projection = output_probabilities

        # Gradient of loss w.r.t. decoder output before final projection
        loss_wrt_norm_ffn = loss_wrt_vocabulary_projection @ decoder.embedding_matrix  # shape: (seq_len, d_model)

        # Gradient of loss w.r.t. the projection matrix (embedding_matrix transposed)
        grads['decoder_layer.embedding_matrix'] = (decoder.norm_ffn.T @ loss_wrt_vocabulary_projection).T

        # ---------------------------------------
        # 2. Decoder FFN normalization layer (LayerNorm after FFN)
        # ---------------------------------------
        (loss_wrt_ffn_layer_2, 
         loss_wrt_norm_cross_att__norm_layer, 
         grads_local) = TransformerHelper.layer_norm_gradient(
            decoder, 
            loss_wrt_norm_ffn, 
            part='ffn'
        )
        grads.update(grads_local)

        # ---------------------------------------
        # 3. Decoder feed-forward network
        # ---------------------------------------
        (loss_wrt_norm_cross_att__ffn_layer, 
         grads_local) = TransformerHelper.feed_forward_gradient(
            decoder, 
            loss_wrt_ffn_layer_2,
            ffn_input_name='norm_cross_att'
        )
        grads.update(grads_local)

        # ---------------------------------------
        # 4. Decoder cross-attention normalization layer
        # ---------------------------------------
        (loss_wrt_cross_att_output, 
         loss_wrt_norm_masked_att__norm_layer, 
         grads_local) = TransformerHelper.layer_norm_gradient(
            decoder, 
            loss_wrt_norm_cross_att__norm_layer + loss_wrt_norm_cross_att__ffn_layer, 
            part='cross_att'
        )
        grads.update(grads_local)

        # ---------------------------------------
        # 5. Decoder cross-attention
        # ---------------------------------------
        (loss_wrt_norm_ffn_encoder_keys_projections, 
         loss_wrt_norm_ffn_encoder_values_projections,
         loss_wrt_norm_masked_att__cross_layer,
         grads_local) = TransformerHelper.multihead_attention_gradient(
             decoder, 
             loss_wrt_cross_att_output, 
             attn_type='cross', 
             input_for_queries=decoder.norm_masked_att,        # decoder Q
             input_for_keys_and_values=encoder.norm_ffn        # encoder K, V
        )        
        grads.update(grads_local)

        # ---------------------------------------
        # 6. Decoder masked attention normalization layer
        # ---------------------------------------
        (loss_wrt_masked_att_output, 
         _, 
         grads_local) = TransformerHelper.layer_norm_gradient(
             decoder, 
             loss_wrt_norm_masked_att__norm_layer + loss_wrt_norm_masked_att__cross_layer, 
             part='masked_att'
        )
        grads.update(grads_local)

        # ---------------------------------------
        # 7. Decoder masked self-attention
        # ---------------------------------------
        (loss_wrt_decoder_pos_encoded_from_keys_projections,
         loss_wrt_decoder_pos_encoded_from_values_projections,
         loss_wrt_decoder_pos_encoded_from_queries_projections,
         grads_local) = TransformerHelper.multihead_attention_gradient(
            decoder, 
            loss_wrt_masked_att_output, 
            attn_type='masked'
        )
        grads.update(grads_local)

        # Total gradient of decoder output embedding (sum of Q, K, V paths)
        grads['decoder_layer.loss_wrt_input_embeddings'] = (
            loss_wrt_decoder_pos_encoded_from_keys_projections + 
            loss_wrt_decoder_pos_encoded_from_values_projections +
            loss_wrt_decoder_pos_encoded_from_queries_projections
        )

        # ---------------------------------------
        # 8. Encoder FFN normalization layer
        # ---------------------------------------
        (loss_wrt_ffn_layer_2, 
         loss_wrt_norm_self_att__norm_layer, 
         grads_local) = TransformerHelper.layer_norm_gradient(
            encoder, 
            loss_wrt_norm_ffn_encoder_keys_projections + loss_wrt_norm_ffn_encoder_values_projections, 
            part='ffn'
        )
        grads.update(grads_local)

        # ---------------------------------------
        # 9. Encoder feed-forward network
        # ---------------------------------------
        (loss_wrt_norm_self_att__ffn_layer, 
         grads_local) = TransformerHelper.feed_forward_gradient(
            encoder, 
            loss_wrt_ffn_layer_2,
            ffn_input_name='norm_self_att'
        )
        grads.update(grads_local)

        # ---------------------------------------
        # 10. Encoder self-attention normalization layer
        # ---------------------------------------
        (loss_wrt_self_att_output, 
         _, 
         grads_local) = TransformerHelper.layer_norm_gradient(
            encoder, 
            loss_wrt_norm_self_att__norm_layer + loss_wrt_norm_self_att__ffn_layer, 
            part='self_att'
        )
        grads.update(grads_local)

        # ---------------------------------------
        # 11. Encoder self-attention
        # ---------------------------------------
        (loss_wrt_encoder_pos_encoded_from_keys_projections,
         loss_wrt_encoder_pos_encoded_from_values_projections,
         loss_wrt_encoder_pos_encoded_from_queries_projections, 
         grads_local) = TransformerHelper.multihead_attention_gradient(
            encoder, 
            loss_wrt_self_att_output, 
            attn_type='self'
        )
        grads.update(grads_local)

        # Total gradient of encoder input embedding (sum of Q, K, V paths)
        grads['encoder_layer.loss_wrt_input_embeddings'] = (
            loss_wrt_encoder_pos_encoded_from_keys_projections + 
            loss_wrt_encoder_pos_encoded_from_values_projections +
            loss_wrt_encoder_pos_encoded_from_queries_projections
        )

        return grads



    @staticmethod
    def layer_norm_gradient(module, loss_wrt_norm_layer_output, part):
        # ----------------------------------------
        # BACKWARD PASS THROUGH LAYER NORMALIZATION
        # ----------------------------------------

        # Retrieve residual input from forward pass (before normalization)
        added_prior_and_residual = getattr(module, f'added_{part}')         # shape: (seq_len, d_model)

        # Get the learned scale parameter (gamma)
        gamma = getattr(module, f'gamma_norm_{part}')                       # shape: (1, d_model)

        # Recompute mean and std from the forward input
        mu = np.mean(added_prior_and_residual, axis=-1, keepdims=True)     # shape: (seq_len, 1)
        sigma = np.std(added_prior_and_residual, axis=-1, keepdims=True) + 1e-6  # add epsilon for stability

        # ----------------------------------------
        # ∂L/∂(input before normalization)
        # Normalize the gradient by reversing: output = gamma * normalized
        loss_wrt_norm_layer_base_input = (loss_wrt_norm_layer_output * gamma) / sigma

        # ----------------------------------------
        # ∂L/∂gamma (scale)
        # Gradient of the loss w.r.t. gamma parameter (for each dimension)
        loss_wrt_gamma = np.sum(
            loss_wrt_norm_layer_output * ((added_prior_and_residual - mu) / sigma),
            axis=0, keepdims=True
        )  # shape: (1, d_model)

        # ----------------------------------------
        # ∂L/∂beta (shift)
        # Gradient of the loss w.r.t. beta parameter (just sum over batch)
        loss_wrt_beta = np.sum(
            loss_wrt_norm_layer_output,
            axis=0, keepdims=True
        )  # shape: (1, d_model)

        # ----------------------------------------
        # Residual gradient is just passed through untouched
        loss_wrt_norm_layer_residual_input = loss_wrt_norm_layer_base_input.copy()

        # ----------------------------------------
        # Package gradients with parameter names for updating
        grads_local = {
            f"{module.__class__.__name__.lower()}_layer.gamma_norm_{part}": loss_wrt_gamma,
            f"{module.__class__.__name__.lower()}_layer.beta_norm_{part}":  loss_wrt_beta,
        }

        return (
            loss_wrt_norm_layer_base_input, 
            loss_wrt_norm_layer_residual_input,
            grads_local
        )



    @staticmethod
    def feed_forward_gradient(module, loss_wrt_ffn_layer_output, ffn_input_name):
        grads_local = {}

        # ----------------------------------------
        # BACKWARD PASS THROUGH FEED-FORWARD NETWORK
        # ----------------------------------------

        # ∂L/∂W_ffn_layer_2
        # Backprop through second linear layer: weight = input^T @ gradient
        relu_ffn = module.relu_ffn                                    # Output from ReLU activation
        loss_wrt_W_ffn_layer_2 = relu_ffn.T @ loss_wrt_ffn_layer_output
        grads_local[f"{module.__class__.__name__.lower()}_layer.W_ffn_layer_2"] = loss_wrt_W_ffn_layer_2

        # ∂L/∂ReLU input (i.e., ∂L/∂layer1 output)
        # Backprop through second weight layer
        W_ffn_layer_2 = module.W_ffn_layer_2
        loss_wrt_relu_ffn = loss_wrt_ffn_layer_output @ W_ffn_layer_2.T

        # ∂L/∂layer1 (before ReLU)
        # Apply gradient of ReLU (mask out values where input was ≤ 0)
        ffn_layer_1 = module.ffn_layer_1
        relu_mask = (ffn_layer_1 > 0).astype(float)
        loss_wrt_ffn_layer_1 = loss_wrt_relu_ffn * relu_mask

        # ∂L/∂W_ffn_layer_1
        # Gradient for first linear layer weights
        ffn_layer_input = getattr(module, ffn_input_name)             # Input to FFN (e.g., norm_cross_att)
        loss_wrt_W_ffn_layer_1 = ffn_layer_input.T @ loss_wrt_ffn_layer_1
        grads_local[f"{module.__class__.__name__.lower()}_layer.W_ffn_layer_1"] = loss_wrt_W_ffn_layer_1

        # ∂L/∂FFN input (to pass further back)
        W_ffn_layer_1 = module.W_ffn_layer_1
        loss_wrt_ffn_layer_input = loss_wrt_ffn_layer_1 @ W_ffn_layer_1.T

        return loss_wrt_ffn_layer_input, grads_local



    @staticmethod
    def multihead_attention_gradient(
        module,
        loss_wrt_attention_layer_output,           # (seq_len, d_model)
        attn_type='masked',
        input_for_queries=None,                    # (seq_len, d_model)
        input_for_keys_and_values=None             # (seq_len, d_model) or None
        ):

        seq_len, d_model = loss_wrt_attention_layer_output.shape
        num_heads = 3
        d_v = d_model   # assumes d_model is already split evenly among heads

        # ----------------------------------------
        # 1. Backprop through final output projection W_output
        # ----------------------------------------

        W_output = getattr(module, f'W_output_{attn_type}_att')

        # Get the output from each attention head from the forward pass
        attention_heads_output = [
            getattr(module, f'output_{attn_type}_att_h_{i}') for i in range(1, num_heads + 1)
        ]

        # Slice W_output by head — each head has its own segment
        W_output_split_by_heads = [
            W_output[i * d_model : (i + 1) * d_model, :] for i in range(num_heads)
        ]  # Each slice: (d_model, d_model)

        # Compute gradient of loss w.r.t. each attention head output
        loss_wrt_output_att_h = [
            loss_wrt_attention_layer_output @ W_out_head.T
            for W_out_head in W_output_split_by_heads
        ]  # Each: (seq_len, d_model)

        # Store gradient w.r.t. W_output (final linear projection)
        concat_heads = np.concatenate(attention_heads_output, axis=-1)
        grads_local = {
            f"{module.__class__.__name__.lower()}_layer.W_output_{attn_type}_att":
                concat_heads.T @ loss_wrt_attention_layer_output
        }

        # ----------------------------------------
        # 2. Backprop through attention output: softmax and V
        # ----------------------------------------

        loss_wrt_values_att_h = []
        loss_wrt_softmax_att_h = []

        for i in range(num_heads):
            softmax = getattr(module, f'softmax_{attn_type}_att_h_{i+1}')
            values  = getattr(module, f'values_{attn_type}_att_h_{i+1}')
            grad_out = loss_wrt_output_att_h[i]

            # ∂L/∂V = softmax.T @ grad_out
            loss_wrt_values = softmax.T @ grad_out

            # ∂L/∂softmax = grad_out @ V.T
            loss_wrt_softmax = grad_out @ values.T

            loss_wrt_values_att_h.append(loss_wrt_values)
            loss_wrt_softmax_att_h.append(loss_wrt_softmax)

        # ----------------------------------------
        # 3. Backprop through softmax (using Jacobian)
        # ----------------------------------------

        loss_wrt_scaled_att_h = []

        for i in range(num_heads):
            softmax = getattr(module, f'softmax_{attn_type}_att_h_{i+1}')
            grad_softmax = loss_wrt_softmax_att_h[i]
            loss_wrt_scaled = np.zeros_like(softmax)

            for idx in range(softmax.shape[0]):
                s = softmax[idx]                      # shape: (seq_len,)
                jac = np.diag(s) - np.outer(s, s)     # Jacobian of softmax
                loss_wrt_scaled[idx] = grad_softmax[idx] @ jac.T

            loss_wrt_scaled_att_h.append(loss_wrt_scaled)

        # ----------------------------------------
        # 4. Backprop through scaling (divide by sqrt(d_k))
        # ----------------------------------------

        loss_wrt_scores_att_h = []
        sqrt_dk = np.sqrt(d_v)

        for i in range(num_heads):
            grad_scaled = loss_wrt_scaled_att_h[i]
            loss_wrt_scores = grad_scaled / sqrt_dk
            loss_wrt_scores_att_h.append(loss_wrt_scores)

        # ----------------------------------------
        # 5. Determine input sources (queries, keys, values)
        # ----------------------------------------

        if input_for_queries is None:
            input_for_queries = getattr(module, 'position_encoded')  # for self-attention

        if input_for_keys_and_values is None:
            input_for_keys_and_values = getattr(module, 'position_encoded')  # self/masked

        # Initialize total input gradients to be accumulated over heads
        loss_wrt_attention_input_for_queries = np.zeros_like(input_for_queries)
        loss_wrt_attention_input_for_keys    = np.zeros_like(input_for_keys_and_values)
        loss_wrt_attention_input_for_values  = np.zeros_like(input_for_keys_and_values)

        # ----------------------------------------
        # 6. Backprop through Q, K, V projections per head
        # ----------------------------------------

        for i in range(1, num_heads + 1):
            # Retrieve forward pass intermediates and weights
            queries = getattr(module, f'queries_{attn_type}_att_h_{i}')
            keys    = getattr(module, f'keys_{attn_type}_att_h_{i}')
            W_queries = getattr(module, f'W_queries_{attn_type}_att_h_{i}')
            W_keys    = getattr(module, f'W_keys_{attn_type}_att_h_{i}')
            W_values  = getattr(module, f'W_values_{attn_type}_att_h_{i}')
            grad_scores = loss_wrt_scores_att_h[i - 1]
            grad_values = loss_wrt_values_att_h[i - 1]

            # ∂L/∂K = Q.T @ ∂L/∂scores
            loss_wrt_keys = queries.T @ grad_scores                   # shape: (d_model, seq_len)

            # ∂L/∂Q = ∂L/∂scores @ K
            loss_wrt_queries = grad_scores @ keys                    # shape: (seq_len, d_model)

            # Backprop into input vectors (Q, K, V)
            loss_wrt_attention_input_for_queries += loss_wrt_queries @ W_queries.T
            loss_wrt_attention_input_for_keys    += loss_wrt_keys.T  @ W_keys.T
            loss_wrt_attention_input_for_values  += grad_values      @ W_values.T

            # Store parameter gradients
            grads_local[f'{module.__class__.__name__.lower()}_layer.W_values_{attn_type}_att_h_{i}']  = input_for_keys_and_values.T @ grad_values
            grads_local[f'{module.__class__.__name__.lower()}_layer.W_keys_{attn_type}_att_h_{i}']    = input_for_keys_and_values.T @ loss_wrt_keys.T
            grads_local[f'{module.__class__.__name__.lower()}_layer.W_queries_{attn_type}_att_h_{i}'] = input_for_queries.T        @ loss_wrt_queries

        # Return input gradients (to propagate backward) and all head parameter gradients
        return (
            loss_wrt_attention_input_for_keys,
            loss_wrt_attention_input_for_values,
            loss_wrt_attention_input_for_queries,
            grads_local
        )
