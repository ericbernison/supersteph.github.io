# GloVE
Glove was an attempt to improve the Word2vec model, previously before the GloVe word embeddings usually had one of the two problems. 
1. It didn’t Utilize the data properly
Or
2. It didn’t learn the relations between the words as shifts
In the case of Word2Vec it gets the relations between the words, but it does not utilize the data in a weighted way. GloVE creates embeddings solely based on the word context pair counts. Let’s start by noting that
<br/>
Xik = the number of times a word appears within the context of I. 
And 
Xi = number of times word i appears in the text corpus
<br/>
Let’s also define 
P(i\j) = Xij/xi, 
aka the probability that a randomly selected occurrence of i will have j as its context.
<br/>
The objective of Glove is to find a function F, F is given 3 inputs, and the objective of F is to return the closeness of two word embeddings. The way we set it up is like this.
F(wi,wj,wk) = P(i\k)/P(j|k)
What does this P(i|k)/P(j|k) mean? Well, we start off with this basic concept again, similar words should appear in similar contexts. In a similar thought, similar words should appear in the same context a similar amount of times, which means that the closer words i and j are in meaning, the closer this () number should be equal to one. If they are far apart however, then this value would either be close to 0 or close to infinity.
 
Since you know that wk is going to be the context matrix, we are going to use a special symbol to denote this.
 
We are trying to find a hypothetical function F that fits these requirements we are going to define it like this F((wi-wj)*ck). The reason that it is wi-wj is because of this: Pik/Pjk is basically the difference between the words i and j, and because vector spaces are linear and can be added, we use the difference between the two vector spaces to eventually find the exact difference.
 <br>
![alt text](https://github.com/supersteph/supersteph.github.io/blob/master/images/Screenshot%20from%202017-06-20%2006-29-15.png "Logo Title Text 1")

Now we reach a problem, we want the context and focus word to be interchangeable, because if the context word appears within the the context of the focus word, if the focus word were the context word, the current focus word could be viewed as the context word.

![alt text](https://github.com/supersteph/supersteph.github.io/blob/master/images/Screenshot%20from%202017-06-20%2006-29-42.png "Logo Title Text 1")
 
We start with this equation
 
![alt text](https://github.com/supersteph/supersteph.github.io/blob/master/images/Screenshot%20from%202017-06-20%2006-30-00.png "Logo Title Text 1")

From this equation we can observe that F is a e^x function, because only e^x functions have the property e^a-b = ea/eb. We can replace the probability functions from the previous equations in replace for F(), and we get

![alt text]( "Logo Title Text 1")
 
We log both sides to get
![alt text](https://github.com/supersteph/supersteph.github.io/blob/master/images/Screenshot%20from%202017-06-20%2006-29-42.png "Logo Title Text 1") 
 
Since xi is not a function of k, and is a constant we can add it to the bias. Then because we are striving to get symmetry, we add the bias of k.
![alt text](https://github.com/supersteph/supersteph.github.io/blob/master/images/Screenshot%20from%202017-06-20%2006-29-42.png "Logo Title Text 1")
 
We are now at the point where we realize that we realize that we don’t want all pairs to be weighted equally, so we use a thing called weighted squares. To compute the loss we go through every I and J and then we weight it based on the number of times it occurs and then square it.
 ![alt text](https://github.com/supersteph/supersteph.github.io/blob/master/images/Screenshot%20from%202017-06-20%2006-30-34.png "Logo Title Text 1")
 
We find an equation f so that f(0) is 0, and that it isn’t weighted too heavily when x is really big
 
Now we use this to get the loss
 ![alt text](https://github.com/supersteph/supersteph.github.io/blob/master/images/Screenshot%20from%202017-06-21%2000-07-31.png "Logo Title Text 1")
 
 
