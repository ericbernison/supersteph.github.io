With the MCTS Algorithim in place, we turn our attention to the model.
Our model is going to take in a game state and return a float estimating the value of that state, and a logit array.

In other words are model is composed of two sub-models a model to get the value and one to get the predictions.

```python
main_input = Input(shape = self.input_dim, name = 'main_input')
#Set a placeholder for the input
x = self.conv_layer(main_input, self.hidden_layers[0]['filters'], self.hidden_layers[0]['kernel_size'])
#Have a conv layer since both sub-models need to detect local features for it to be successful
if len(self.hidden_layers) > 1:
	for h in self.hidden_layers[1:]:
		x = self.residual_layer(x, h['filters'], h['kernel_size'])

vh = self.value_head(x)
ph = self.policy_head(x)

model = Model(inputs=[main_input], outputs=[vh, ph])
model.compile(loss={'value_head'+str(self.version_number): 'mean_squared_error', 'policy_head'+str(self.version_number): softmax_cross_entropy_with_logits},
	optimizer=SGD(lr=self.learning_rate, momentum = config.MOMENTUM),	
	loss_weights={'value_head'+str(self.version_number): 0.5, 'policy_head'+str(self.version_number): 0.5}	
	)

```

The model is set up like the following.

## Why Deep Networks
Hypothetically, having a model that is one hidden layer would be able be able to produce similar results to any Deep Networks, so why do it?

There are multiple reasons for this, first, while it is mathmatically possible. By creating a wide models it ends up with a function that is approximating the real function rather than getting the smooth function.


By creating a function with multiple layers that are pooled as you get to the top you'd also be able to address different things in different layers.

For example, at the bottom layer of a model, what you are concerened with is the movement of the pawns.	And perhaps, at the higher level, you would be worrying about how the different pieces protect each other.

## Our Model
With the logic to use deep networks to abstract things. Our model consists of a stack of Convolutional layers using pooling.

```python
def conv_layer(self, x, filters, kernel_size):

	x = Conv2D(
	filters = 75
	, kernel_size = (4,4)
	, data_format="channels_first"
	, padding = 'same'
	, use_bias=False
	, kernel_regularizer = regularizers.l2(self.reg_const)
	)(x)

	#have 75 4x4 kernels form the convolution layer

	x = BatchNormalization(axis=1)(x)
	x = LeakyReLU()(x)
	#Activation Function that is needed in Deep Networks

	return (x)
```

However, with deeper models, there is a new problem introduced: Gradient vanishing. This concept happens because the gradients have to travel a large distance. So much of the original meaning of the gradient is lost. To combat this we introduce a layer called a resiudal layer. By adding the input of the current layer to the output, we are able to keep the original meaning of the gradients throughout the layers, because it is added directly to the output of the layer.


```python
x = self.conv_layer(input_block, filters, kernel_size)	

x = Conv2D(
filters = filters
, kernel_size = kernel_size
, data_format="channels_first"
, padding = 'same'
, use_bias=False
, activation='linear'
, kernel_regularizer = regularizers.l2(self.reg_const)
)(x)

x = BatchNormalization(axis=1)(x)

x = add([input_block, x])

x = LeakyReLU()(x)

return (x)
```