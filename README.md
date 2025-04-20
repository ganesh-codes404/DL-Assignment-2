# Deep Learning Assignment-2 
Roll No: 1601-22-737-176


**There is some sort of issue with the notebook of the 2nd Question, this is the notebook link( https://colab.research.google.com/drive/1tdQ72kegxUVTS5BjlYDbN38qOwMCDUfM?usp=sharing )** 

The Assignment has 2 questions, one deals with the Seq2Seq model to implement and train the model on the Dakshina Dataset, the other question deals with fine tuning the GPT-2 Model to generate lyrics(used HuggingFace to utilize the model)

## Question 1
For the first question, Dakshina Dataset has to be downloaded and then uploaded to the Google Drive and it can accessed from there by the following code:

```python 
from google.colab import drive
drive.mount('/content/drive')

file_path = '/content/drive/MyDrive/dakshina_dataset_v1.0.tar' 
```
From here on we can build our model and answer the quesitons, in code the it is a sequence‑to‑sequence (seq2seq) encoder–decoder architecture, where:

**Encoder** \
Takes your input token sequence \
Embeds each token into a dense vector of size embedding_dim \
Feeds the embeddings into an RNN layer (by default an LSTM) to produce a final hidden (and, for LSTM, cell) state \
**Decoder** \
Takes the target token sequence (shifted one step) as input \
Embeds each token into the same-sized vectors \
Feeds these, along with the encoder’s final state(s), into another RNN layer (again, by default an LSTM) to produce a sequence of decoder hidden states \
Projects each decoder hidden state through a Dense+softmax over your output vocabulary \
Because you call build_model(cell_type='LSTM',…), you’re using LSTM cells in both the encoder and decoder. If you passed cell_type='GRU' or 'SimpleRNN', it would swap to those, instead, but the overall pattern remains a standard encoder‑decoder with an embedding layer → RNN layer → dense output. \

**More than 2 Lakh computations and trainable parameters will be there**

## Question 2
The dataset is from Kaggle https://www.kaggle.com/paultimothymooney/poetry, the process for reading the dataset is the same as the previous question

All poem/lyric files are read into a single list and used to create a Hugging Face Dataset.

Model and Tokenization

Model: GPT-2 from Hugging Face's model hub.
Tokenizer: GPT2Tokenizer with the pad_token set as the eos_token (since GPT-2 doesn't include a padding token by default).
The model is fine-tuned using masked language modeling with tokenized lyric text.

Running the Script

Ensure you have the required libraries installed.
Run the script in a cloud-based platform with GPU enabled so that it runs very quickly.

## Conclusion

This Assignment demonstrates fine-tuning GPT-2 to generate song lyrics based on a given prompt, as well as the training of a Seq2Seq model for sequence generation tasks using the Dakshina dataset.


