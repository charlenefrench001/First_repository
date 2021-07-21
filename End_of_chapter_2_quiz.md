# First_repository
Linking Github and RStudio

1. What is the order of the language modeling pipeline?
a. First, the model, which handles text and returns raw predictions. The tokenizer then makes sense of these predictions and converts them back to text when needed.
b. First, the tokenizer, which handles text and returns IDs. The model handles these IDs and outputs a prediction, which can be some text.
c. The tokenizer handles text and returns IDs. The model handles these IDs and outputs a prediction. The tokenizer can then be used once again to convert these predictions back to some text.
Correct! Correct! The tokenizer can be used for both tokenizing and de-tokenizing.

2. How many dimensions does the tensor output by the base Transformer model have, and what are they?
The sequence length, the batch size, and the hidden size

3. Which of the following is an example of subword tokenization?
WordPiece
Correct! Yes, that's one example of subword tokenization!
Character-based tokenization
Splitting on whitespace and punctuation
BPE
Correct! Yes, that's one example of subword tokenization!
Unigram
Correct! Yes, that's one example of subword tokenization!


4. What is a model head?
A component of the base Transformer network that redirects tensors to their correct layers
Also known as the self-attention mechanism, it adapts the representation of a token according to the other tokens of the sequence
An additional component, usually made up of one or a few layers, to convert the transformer predictions to a task-specific output
Correct! That's right. Adaptation heads, also known simply as heads, come up in different forms: language modeling heads, question answering heads, sequence classification heads... 

5. What is an AutoModel?
A model that automatically trains on your data
An object that returns the correct architecture based on the checkpoint
Correct! Exactly: the AutoModel only needs to know the checkpoint from which to initialize to return the correct architecture.
A model that automatically detects the language used for its inputs to load the correct weights
Incorrect. Incorrect; while some checkpoints and models are capable of handling multiple languages, there are no built-in tools for automatic checkpoint selection according to language. You should head over to the Model Hub to find the best checkpoint for your task!

6. What are the techniques to be aware of when batching sequences of different lengths together?
Truncating
Correct! Yes, truncation is a correct way of evening out sequences so that they fit in a rectangular shape. Is it the only one, though?
Returning tensors
Incorrect. While the other techniques allow you to return rectangular tensors, returning tensors isn't helpful when batching sequences together.
Padding
Correct! Yes, padding is a correct way of evening out sequences so that they fit in a rectangular shape. Is it the only one, though?
Attention masking
Correct! Absolutely! Attention masks are of prime importance when handling sequences of different lengths. That's not the only technique to be aware of, however.


7. What is the point of applying a SoftMax function to the logits output by a sequence classification model?
It applies a lower and upper bound so that they're understandable.
Correct! Correct! The resulting values are bound between 0 and 1. That's not the only reason we use a SoftMax function, though.
The total sum of the output is then 1, resulting in a possible probabilistic interpretation.
Correct! Correct! That's not the only reason we use a SoftMax function, though.

8. What method is most of the tokenizer API centered around?
Calling the tokenizer object directly.
Correct! Exactly! The __call__ method of the tokenizer is a very powerful method which can handle pretty much anything. It is also the method used to retrieve predictions from a model.

9. What does the result variable contain in this code sample?

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
result = tokenizer.tokenize("Hello!")

A list of strings, each string being a token
#codes to pass a model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "Hello!"

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

batched_ids = torch.tensor([ids])
print("Input IDs:", batched_ids)

output = model(batched_ids)
print("Logits:", output.logits)


10. Is there something wrong with the following code?

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModel.from_pretrained("gpt2")

encoded = tokenizer("Hey!", return_tensors="pt")
result = model(**encoded)

The tokenizer and model should always be from the same checkpoint.
Correct! Right!
It's good practice to pad and truncate with the tokenizer as every input is a batch.
Incorrect. It's true that every model input needs to be a batch. However, truncating or padding this sequence wouldn't necessarily make sense as there is only one of it, and those are techniques to batch together a list of sentences.
