# Point of Lasso
We have a very great idea of how to get a function given linear data, but what happens if we do not know the type of function that best fits the data?
For example if we had data like this.
[picture of weird data]
In many cases, we don’t know the most efficient way to map x<sub>i</sub> to A<sub>i</sub>. Making A a larger vector with additional mappings sounds like a tempting idea at first, after all more weights should lead to a more concise model. In reality, however, the larger Ai leads to overfitting and slower models.

If we leave our model with the overfitted weights it would look like this
[picture of overfitted thing]
We don’t like this because we do not actually learn the trends, our model just memorizes the points.

# Increase Accuracy
 To increase the accuracy of the model, we introduce a penalty that encourage the weights to either be small, or approach zero. While this initially does not seem to solve our problems, it is an elegant and efficient way to deal with both of them

# Overfitting
Our data set is going to be naturally noisy, the data points that we generate aren’t naturally all going to be on the function that we are trying to find. To get a function that is noisy, we would need very large coefficients. This becomes undesirable because when we are copying the noise, the model isn’t learning the trends instead it is just copying the data. Because we want our model to learn the trends, we lower the coefficients.

# Unused terms
As we previously explained, the x<sup>2</sup> term makes the entire set more spread out, we use this intuition to do the opposite: We use the x term to force some values to be 0. When we apply this techniques, the variables that aren’t important for the model are going to approach zero. Once they get really close to zero the weight is negligible and they can be removed for a smaller model.

Lasso introduces a variable called the Lasso Penalty, which looks like this

<img src="/images/CodeCogsEqn (1).gif">
The t in this equation corresponds to the amount of regularization that our model does. By adding together all elements in our β vector and trying to make it less than a certain value, we encourage our model to learn smaller variables.
