## Introduction

This project is a simple implementation of the **original Transformer model**, from the paper  
*Attention Is All You Need* (Vaswani et al., 2017).

The code is written in **Python using NumPy only**. It does not use any deep learning libraries like PyTorch or TensorFlow.

The main goal of this project is to help understand how the Transformer model works inside.  
Everything from the forward pass to training with gradient descent is built completely **from scratch** using only basic NumPy operations.

Both the forward computations and the gradient-based learning are implemented manually, without any automatic differentiation.

The model was tested on a small, simple dataset, where it successfully showed that it can learn.  
This makes it a great example for learning how Transformers work at a low level.


---

### Main features

This Transformer includes all parts from the original model:
- Learning by gradient descent
- Encoder-Decoder structure  
- Multi-Head Attention  
- Scaled Dot-Product Attention  
- Feed-Forward Networks  
- Positional Encoding  
- Residual connections and Layer Normalization  

To make it easier to understand, here are some diagrams that show the main ideas:

#### Transformer Architecture

This image shows how the encoder and decoder work together in the model.

![Encoder-Decoder Architecture](./readme_images/architecture.svg)

#### Multi-head attention

Each attention layer has several heads. Each head looks at the input in a different way.

![Multi-Head Attention](./readme_images/multi_head_attention.svg)

#### Scaled dot-product attention

This image shows how queries, keys, and values are used to calculate attention.

![Scaled Dot-Product Attention](./readme_images/scaled_dot_product_attention.svg)







## Implementation

This Transformer is implemented inside a **Jupyter Notebook**.

The only library used is **NumPy**. There is no PyTorch, TensorFlow, or any deep learning framework.

The code is **modular**; split into different classes and helper functions to keep it organized and easy to follow.

---

### Simplifications

To reduce the amount of computation and make things easier to understand, some parts of the model are simplified. These changes do not affect the structure of the Transformer.

| Component           | Original Paper | This Version |
|---------------------|----------------|--------------|
| Encoder Layers      | 6              | 1            |
| Decoder Layers      | 6              | 1            |
| Attention Heads     | 8              | 3            |
| `d_model`           | 512            | 4            |
| `d_k`, `d_v`        | 64             | 4            |
| Feed-Forward (`d_ff`)| 2048          | 4            |

- Attention heads are **not computed in parallel**. Instead, each head is processed **one at a time**, to make the logic easier to understand.
- Even with these simplifications, the **full Transformer structure** is implemented.
- The model can **learn from simple data**, which is enough for testing how learning works.
- It is **easy to scale** this code to match the full Transformer, but that was not the goal of this small educational project.

---

## Architecture and classes

This diagram shows how different classes are connected inside the implementation:

![Class Diagram](./readme_images/class_diagram.svg)

### Class overview

- `Transformer`: the main class that manages the training and model logic.
- `Encoder` and `Decoder`: separate classes for encoder and decoder blocks.
- `TransformerHelper`: static methods that manage forward and backward steps.

---

## How the code was implemented

The Transformer was built step by step, starting from the forward pass and then adding gradients for training.

### Forward pass

- Built each component one by one
- Tested encoder, decoder, and attention parts to make sure shapes and logic were correct

### Backward pass (gradients)

- Gradients were **calculated manually** using NumPy
- Each part has its own gradient function
- Matrix shapes were printed during testing to verify the chain rule and backpropagation

#### Gradient flow diagram

![Gradient Flow](./readme_images/derivatives.svg)

---

## Training and testing

- The model was trained on a **very simple dataset**
- A full **training loop** was implemented to calculate loss and update parameters
- Even with small dimensions, the model was able to **learn and reduce the loss**
- This proves the implementation works correctly

---

## Conclusion

This project shows how a Transformer works **at a low level**, without any hidden operations.

It is made for **learning purposes**, and helps anyone understand how attention and the Transformer architecture function.

The code can be easily extended to **larger models or real-world datasets**, but this small version is enough to learn the core ideas.
