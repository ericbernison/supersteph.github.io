<h1>Usefulness and Applications</h1>
Word2vec has multiple applications, in standard machine learning context it is very tough to retain information from a single word unless more information is given. In most forms are data, for a standard sized image there could be thousands of bytes of information, but for words this is not the case. It is very simple to imagine each word as one particular number, making the total information that the sentence contains minisculey small compared to the image that we may imagine to have the same amount of information. For example, from a machine’s perspective when it hears the word “king” it doesn’t know anything about it, but when we hear that word we automatically associate it with royalty and male. Word2vec essentially gives the word more information and gives the machine a sort of “intuition” into what a word means.
<br/><br/>
For and example of how this kind of works with our mind let's say that we said "a cat riding a toy horse" an image like this might pop up in your mind <br/> <br/>
 
 
![alt text](https://github.com/supersteph/ro_sgns/blob/master/images/Lr7J8ab.jpg "cat on a horse")
 
 
However, the data obtained by strictly the words in the sentence is 22 bytes, while the image itself is 18 kb. The image almost contains times the information that the sentence does.<br/>
 
By using word2vec the information is turned from a sparse vector into a dense one, and this allows the machine to do things with it.<br/>
 
 <h2>Skip-Gram Model</h2>
 
Word2vec is essentially a type of unsupervised learning, but it is trained with a fake supervised learning task. It starts with the basic idea that words that appear in similar context will have similar meanings. With this idea in mind we try to to maximize the probability of the context word from the actual word. We first create a fake model that has one goal: predict the context from a current word. The context is defined as the words surrounding the current word, everything within a certain number of words of the current word would be it's context. The length from the focus word to the farthest context can be varied to make our model better.
<br/> 
This fake model is made up of two layers: an embedding layer and then a output layer with the context embeddings. The embedding layer will be size [vocab_size,embedding_size]. We feed this model a focus word, we will use a one hot encoding to express this word. The input to this layer will be a one hot encoding of size [vocab_size]. The product of a one hot encoding and a rectangular matrix will be a single of row of that matrix. We call this process an embedding lookup, because it's almost like looking up something from a dictionary.
<br/>
 
![alt text](https://github.com/supersteph/ro_sgns/blob/master/images/word2vec_weight_matrix_lookup_table.png "Layers")
 
 
The output of this layer is often called the context embedding. And the result of the product with the word embeddings is labeled X. More things are done with X, but for simplicity sake, we can assume that X is a scalar right now and we use it to get the probability. after the product all that remains is a softmax layer. This softmax layer computes the probability as follows, it first gets the scalar product and puts it through the exp function and then it is divided by the sum of up all the scalars. This forms an output that if every single element is summed together would equal to one.
 
<br/>
 
<h1>Clever Tricks</h1>
 
 
 
This basic model works well but there are several downfalls to the basic mode, it is very slow and it takes a lot of time to train. It requires a lot training data and is all-around very hard to do. 
 
## Phrasing
The main idea behind phrasing is that certain sequence of words should be treated as one word, and words of certain caliber could be treated not as phrases but as single words. A prime example of this disparity between the addition of words is the word “Boston” and the word “Globe” these words combined is a newspaper and should be treated like a different object that a globe that comes from boston. The phrasing algorithim is not going to be gone over in this blog but if you want to read on it the link is over [here](http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/)
<br/><br/>
The adding of phrases turns the already big vocabulary humongous but ultimately allows for the model to have a better understanding of the language. 
## SubSampling
The main purpose of SubSampling is to remove words from the context that do not provide any meaning to the current word, or appear too frequently. As it turns out, words that don’t provide much meaning and appear too frequently are often the same words. For example, the word “the” the only new information captured by the word “the” is that perhaps the word following is a noun. However “the” does not have to appear the amount of times that it does for us to get a good embedding from it. So we decide to remove some of these words that appear very frequently from the context, which provides us with just enough information to embed these words. The probability of removing a particular word from the context is a function with the frequency as a variable. This function removes words that appear more frequently with higher probability. The probability is modeled by the function
<br/>
(o/0.001+1)*0.001/o
<br/>
Where o stands for the number of occurrences of that word over the total words.
 
## Negative Sampling
Negative Sampling is used to generate a random sample which you label the weight zero and then it works better and creates more random data. Negative Sampling usually takes 5 words and apply negative weights to these five words. The point of this is to minimize the weights that you update, if you put negative weights on everything other than your current word two things happen, you are updating the entire model every single training iteration which leads to training being extremely slow. You are also doing some work in the opposite way because you might be labelling things negative that may appear in other contexts. For example, just because the word “person” appears in the context of the word “fat” it does not mean that “person” can’t also appear in the context of the word “tall”. On the other hand if you do not provide any other negative weights your training ends up very slow once again, and the model might be able to train things you know aren’t supposed to appear in that context.You find a bunch of words that aren’t your current word and say that you don’t want those to show up at all this means that you don’t have to update all the weights the entire time. Instead of updating everything at one time by only putting some of the things as negative it allows the model to update the output layer much faster, while the embedding layer is unchanged from the previous version. The reason you want to pick the words that come more frequently is logic similar to the subsampling idea, the more often a word appears, the more likely it is to be a filler word and have no correlation to the current word. 
<br/>
The loss function using Negative Sampling looks like this
<br/>
![alt text](https://github.com/supersteph/ro_sgns/blob/master/images/equation.png "negative sampling loss equation")
<br/>
we use this expression to replace ever log() in the loss function for the skip gram model. The first part of this equation can be known as the original part, but then the negative samples are added and then it is like that. The k in the equation stands for the sampling rate, this sampling rate is a function of the frequency of the word.
<br/>
the sampling equation is the probability of the current word to the power of 3/4 divided by the sum of all the probabilities of the context words to the power of 3/4.
