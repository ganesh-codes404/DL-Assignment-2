# Deep Learning Assignment-2 
Roll No: 1601-22-737-176

## Question 1 
Build an RNN based seq2seq model, which contains the following layers: (i) input layer for character embeddings (ii) one encoder RNN, which sequentially encodes the input character, sequence (Latin) (iii) one decoder RNN which takes the last state of the encoder as input and produces one output character at a time (Devanagari).

(1) What is the total number of computations done by your network? (assume that the input embedding size is mmm, encoder and decoder have 1 layer each, the hidden cell state is k for both the encoder and decoder, the length of the input and output sequence is the same, i.e., T, the size of the vocabulary is the same for the source and target language, i.e., V)

A) More than 2L computations.

(2) What is the total number of parameters in your network? (assume that the input embedding size is mmm, encoder and decoder have 1 layer each, the hidden cell state is k for both the encoder and decoder and the length of the input and output sequence is the same, i.e., T, the size of the vocabulary is the same for the source and target language, i.e., V)

A) More than 2L trainable parameters.

(3) Use the best model from your sweep and report the accuracy on the test set and Provide sample inputs from the test data and predictions made by your best model.

A) Can be seen in the colab notebook.

## Question 2 
Your task is to finetune the GPT2 model to generate lyrics for English songs. You can refer to (https://towardsdatascience.com/natural-language-generation-part-2-gpt-2-and- huggingface-f3acb35bc86a ) and follow the steps there. This blog shows how to fine-tune the GPT2 model to generate headlines for financial articles. Instead of headlines, you will use lyrics so you may find the following datasets useful for training: https://data.world/datasets/lyrics
https://www.kaggle.com/paultimothymooney/poetry

### Dataset
The dataset used consists of .txt files containing English poems (https://www.kaggle.com/paultimothymooney/poetry). Only files with more than 50 characters are considered viable. The files are read from a local directory (/content/drive/My Drive/archive-2) and stored in a pandas DataFrame.

### Tokenization

We use the GPT-2 tokenizer from HuggingFace’s transformers library. Padding and truncation are applied to each input sequence to ensure a uniform length of 256 tokens. The tokenizer uses the EOS token as the padding token to maintain GPT-2 compatibility.

### Model Training

A GPT2LMHeadModel is initialized with pretrained weights from HuggingFace's gpt2. The model is fine-tuned using the Trainer API.

Key training parameters:

Number of epochs: 3
Batch size per device: 2
FP16 training enabled if GPU is available
Output directory: ./gpt2-lyrics
Data collator: Language modeling collator with mlm=False (causal LM)
The model is trained on the tokenized poetry dataset to learn patterns in lyrical text.

### Text Generation

After training, we input a prompt like:
"When the sun rises and stars begin to fade"

The model generates a lyrical continuation using top-k sampling and nucleus (top-p) filtering. Generation parameters:

max_length: 100
do_sample: True
top_k: 50
top_p: 0.9
temperature: 1.0
repetition_penalty: 1.2



