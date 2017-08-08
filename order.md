#Order Embeddings

Conventional Embeddings have had trouble showing relationships such as hypernymy. Order Embeddings are the attempt to fix that. Order embeddings have been able to show state of the art performance in detecting hypernyms and show promising results in other tasks such as textual entailment and image captioning. All these tasks have something in common: there is a latent hierarchy, and if you just embed the latent hierarchy into an standard embedding space you won’t be able to observe the hierarchy, that’s why we use order embeddings.


##Hypernyms
In this blog we will mainly be focusing on the implications and applications in hypernymy detection. Hypernymy is defined as the “is a” relationship between words. By definition, hypernymy is asymmetrical in the case that if “color” is the hypernym of “red”, “red” is not the hypernym of “color”. In hypernymy, the structure exists in the way that words that are hypernyms are going to exist in a “higher” space than their corresponding hypnoymns. In other words, words like “pig” are going to be in a higher embedding space than words like “piglet”. Since “piglet” is a pig but pig is not a type of piglet. 

##Partial Order Structure
This leads us to the question how do we model this relationship. However when we think about this embedding space, how are words that have seemingly no relationship with the word pig going to be expressed? For example, you can’t definitely rank the word “red” in front of the word piglet. That’s why we can assume that the hypernymy structure is going to be a partial order structure. A partial order by definition is a set. This set has a structure, It looks like this. Each of the elements in this set could have a relationship with another element. This relationship is going to be a transitive and antisymmetric relationship, which in our case it means that this relationship means is higher than. Since they both can’t be higher it can’t be antisymmetric, and then this also means that if a is higher than b and b is higher than c a is definitely going to be higher than c. So there are no loops. This structure fits our purposes because of the property that every element doesn’t have to be comparable with each other, in some cases you can’t tell if d is higher than e or if e is higher than d.

##How Do We Design the Hierarchy?

Bringing this to embeddings you define higher as each value in the vector being higher than the other one. There is going to be one big embedding space, the reasoning behind this is that you want the space to be differentiable, and if you have separate embedding spaces the jumps between the spaces would make it very hard to apply gradient descent, that’s why when the paper says multiple embedding spaces it really means one big embedding space. Now we have the problem of what makes a certain word higher than the other word? In this paper it defines a word being higher as given the vector value, if each value is higher than the corresponding value in the compared hyponym then you can comfortably say that that word is a hypernym of the other word. This works well because you are going to have the higher structure, it also works in a way since each word isn’t going to necessarily be higher than each other.

##Setup
The paper trains the model with part of the WordNet dataset. Wordnet is a large dataset that maps out a lot of relationships between words, in this case hypernymy. We set up the data into two parts a postive set, where a pair of words have a hypernymy relationship. Then we set up a negative set, where the pair of words have no discernable hypernymy relationship. We then design the loss function so that the postive examples modeled to be higher in each value, and the negative examples are discouraged from being on a higher level.
##Ending Thoughts
This approach has promise, and I beleive that the fundamental idea can be applied to other places to allow it to be better. But for hypernymy prediction I believe it is still a bit lacking due to the fact that it has to be trained knowing the hierarchy.
###Source
[Original Paper](https://arxiv.org/pdf/1511.06361.pdf)
