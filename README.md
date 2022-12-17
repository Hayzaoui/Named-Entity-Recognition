# Named-Entity-Recognition

The project involves building and optimizing models for Named Entity Recognition in text using real data. The goal is to recognize entities, such as names or organizations, in a given piece of text. This is a common task in natural language processing and can be useful for a variety of applications, such as information extraction or text classification.

In the project, two models are implemented to solve this task. The first model is a simple model, which uses Support Vector Machine. The second model is a feedforward neural network, which takes as input a sequence of word embeddings and passes them through a series of fully connected linear layers with non-linear activations. 

The models are trained on word embeddings obtained from pre-trained vectors, GloVe of tweets. In the second model, additional binary features are added to the word embeddings to improve the model's performance. The models are evaluated using the F1 score, which is a metric that measures the balance between precision and recall in a binary classification task. The F1 score is calculated by chaining the predictions of the model and the real labels across all sentences and comparing the two lists. 

Overall, the project involves implementing and optimizing models for Named Entity Recognition, performing language processing tasks on real data, and analyzing the nature of the success. The models are trained on word embeddings and evaluated using the F1 score to measure their performance.



