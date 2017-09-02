# Introduction to Regression
You have just quit your job, you are faced with the problem with whether you want to go to school or whether you should go directly to work. You decide that you should gather data to find out the relationship between the amount of education versus the salary.

## Regression

To find this function we use a technique called regression. This technique, at its core, is about finding the best function to fit the data that is provided. In the next part we go through the lasso regression technique and in the future we show how it can be applied to more complex models to solve tasks such as digit recognition.


Given our data
<br/>
![alt text](/images/Screenshot%202017-08-31%20at%2010.27.01%20AM.png "data")
<br/>
We are trying to find a function like this
<br/>
![alt text](/images/Screenshot%202017-08-31%20at%205.13.52%20PM.png "data with line")
<br/>
As you can see, this line minimizes the distance between each point and the line.

## Least squares regression

We first introduce the most basic type of regression: least square regression. This approach, allows us to find the desired function.

The basic idea is that we are trying to find the ß’s (coefficients) that allow this function to be as close to our data as possible
<br/>
β<sub>0</sub>+β<sub>1</sub>*x<sub>i</sub>


## Objective Function

We now want to find out  how we can measure how “good” our function is. We define the closeness of the function using this expression.
<br/>
![alt text](/images/CodeCogsEqn.gif "equation") 
<br/>
This value can be thought of as the loss of that function. The objective of our model is to minimize this loss. Since β<sub>0</sub>+β<sub>1</sub>*x<sub>i</sub> is our function y<sub>i</sub>-β<sub>0</sub>+β<sub>1</sub>*x<sub>i</sub> is going to be a measure of how close our function is to the actual value. 
The reason why we square the  y<sub>i</sub>-β<sub>0</sub>+β<sub>1</sub>*x<sub>i</sub> expression is for two reasons 
1. We want all the differences to be the same sign so that when we add them together the signs won’t cancel each other out. 
And  
2. We want our function be close to every single point instead of being on one point and being far from all of the other points. When the squared function is used, larger values are now going to be exponentially larger, therefore (additionally) discouraging large outliers. By adding the squared term, it is now more efficient to have all of our values consistently close to the actual function rather than having our model hit a point and be completely off basis for other points.

## Expansion
Our model is solely responsible for finding the coefficients in our function; we are not concerned with the function that we plug our coefficients into. When we have a functions that are not linear we need syntax for it.
For example, if our function looks like this
</br>
![alt text](/images/nmpMg.png "non linear") 
<br/>
Our function is obviously going to try and find the linear way of getting this when this function is obviously not linear.
We design a structure that can support functions that aren’t only linear.
A, a vector of size N*m, is going to represent the outline of the function. Each element in A is a separate vector that depends solely on the value of the ith input (x<sub>i</sub>). The process of mapping an x<sub>i</sub> to the corresponding A<sub>i</sub> is the same process every time. Given the x<sub>i</sub> there are going to be m functions that you put it through, and each result is kept separate to form an m sized vector: A<sub>i</sub>.
β, a vector of size m, is going to be our inputs.

In the our case our Ai vector would look like this {1,x<sub>i</sub>}. Our β is going to be in the form of {β<sub>0</sub>,β<sub>1</sub>}. Once we dot product these two vectors together it is going to form the expression that we are trying to optimize.


 
## Optimization
Since the objective function is an even function, the minimum value is going to be at one of the vertices the place where the gradient is equal to zero. In standard statistics the gradients are readily available and you can get the betas by finding the gradient and setting it to zero. 
When the gradients aren’t readily available, however, we need a different solution. This solution is gradient descent and through many iterations we can eventually find the betas that minimize the objective function.
