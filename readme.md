# Model Input Parameters Documentation

This document provides details on the input parameters used in a PyTorch-based transformer model.

## 📌 Input Parameters

### 1️⃣ `input_ids` (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
Indices of input sequence tokens in the vocabulary. Padding is ignored by default.

🔹 **How to obtain:** Use [`AutoTokenizer`].  
🔹 See: [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`].  
🔹 [What are input IDs?](../glossary#input-ids)

---

### 2️⃣ `attention_mask` (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
Mask to prevent attention on padding token indices.

🔹 Values:
- `1` → Token is **not masked**
- `0` → Token is **masked**

🔹 **How to obtain:** Use [`AutoTokenizer`].  
🔹 See: [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`].  
🔹 [What are attention masks?](../glossary#attention-mask)

📌 **Additional Notes**:
- If `past_key_values` is used, only the last `input_ids` need to be provided.
- Modify [`modeling_opt._prepare_decoder_attention_mask`] for custom padding behavior.
- See [this paper](https://arxiv.org/abs/1910.13461) for default strategy details.

---

### 3️⃣ `position_ids` (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*)
Specifies positions of input tokens in the position embeddings.

🔹 Range: `[0, config.n_positions - 1]`  
🔹 [What are position IDs?](../glossary#position-ids)

---

### 4️⃣ `past_key_values` (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*)
Stores precomputed key-value states for faster sequential decoding.

🔹 **Formats Supported:**
1. [`~cache_utils.Cache`] instance ([KV Cache Guide](https://huggingface.co/docs/transformers/en/kv_cache))
2. Tuple format:

🔹 The model outputs the same cache format that is fed as input.  
🔹 If `past_key_values` are used, only the last `input_ids` need to be provided (`(batch_size, 1)`).  

---

### 5️⃣ `inputs_embeds` (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*)
Allows direct input of embedded representations instead of `input_ids`.

📌 **Use Case:**  
Useful when customizing how token indices are converted to embeddings.

---

### 6️⃣ `use_cache` (`bool`, *optional*)
- `True`: Returns `past_key_values` to accelerate decoding.  
- `False`: No caching.

---

### 7️⃣ `output_attentions` (`bool`, *optional*)
- `True`: Returns attention tensors for all layers.  
- `False`: Does not return attention values.

📌 See `attentions` under returned tensors for details.

---

### 8️⃣ `output_hidden_states` (`bool`, *optional*)
- `True`: Returns hidden states for all layers.  
- `False`: Does not return hidden states.

📌 See `hidden_states` under returned tensors for details.

---

### 9️⃣ `return_dict` (`bool`, *optional*)
- `True`: Returns a [`~utils.ModelOutput`] object.
- `False`: Returns a plain tuple.

---

### 🔟 `cache_position` (`torch.LongTensor` of shape `(sequence_length)`, *optional*)
Indicates the position of input tokens in the sequence.

📌 **Difference from `position_ids`**:
- `cache_position` is **not affected by padding**.
- Used to update the cache in the correct position and infer sequence length.

---

## 📖 References
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [KV Cache Guide](https://huggingface.co/docs/transformers/en/kv_cache)
- [Attention Masks](../glossary#attention-mask)
- [Position IDs](../glossary#position-ids)
