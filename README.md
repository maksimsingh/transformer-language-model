ransformer Language Model

Overview



This project implements a character-level Transformer language model from scratch using PyTorch.



The model learns to predict the next character in a sequence and can generate new text based on the style of the training data.



Features: 



Multi-head self-attention

Positional embeddings

Autoregressive text generation

Training and validation loss tracking



The project includes two datasets:



input1.txt — text written in a Shakespeare-like style

input2.txt — collection of jokes



This allows the model to generate text in different styles depending on the input.



How it works



The model:



1\.Encodes characters into embeddings

2\.Adds positional information

3\.Processes sequences through Transformer blocks

4\.Predicts the next character using probability distribution



Usage



Run the model with a specific dataset: python language\_model.py input1.txt or python language\_model.py input2.txt



If no file is provided, the default file will be used.



Example Output



After training, the model generates new text based on the dataset style.



Example (trained on Shakespeare-style text):



Thou art the light that shines upon the soul,

And whispers truth where silent shadows fall...



Tech Stack: 



Python

PyTorch



Project Structure:



transformer-language-model/

│

├── language\_model.py

├── input1.txt

├── input2.txt

├── requirements.txt

└── README.md



Requirements



Install dependencies:



pip install -r requirements.txt



Notes:



This project is implemented for educational purposes to understand how Transformer architectures work internally without relying on high-level abstractions.

