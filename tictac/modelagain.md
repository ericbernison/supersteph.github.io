#Model
With the MCTS Algorithim in place, we turn our attention to the model.
Our model is going to take in a game state and return a float estimating the value of that state, and a logit array.

In other words are model is composed of two sub-models a model to get the value of the current state and one to get the predictions for the actions that should be taken.

```python
main_input = Input(shape = self.input_dim, name = 'main_input')
#Set a placeholder for the input
x = self.conv_layer(main_input, self.hidden_layers[0]['filters'], self.hidden_layers[0]['kernel_size'])
#Have a conv layer since both sub-models need to detect local features for it to be successful
if len(self.hidden_layers) > 1:
	for h in self.hidden_layers[1:]:
		x = self.residual_layer(x, h['filters'], h['kernel_size'])
#get the value and policy
vh = self.value_head(x)
ph = self.policy_head(x)

#set the model as a dual output
model = Model(inputs=[main_input], outputs=[vh, ph])
model.compile(loss={'value_head'+str(self.version_number): 'mean_squared_error', 'policy_head'+str(self.version_number): softmax_cross_entropy_with_logits},
	optimizer=SGD(lr=self.learning_rate, momentum = config.MOMENTUM),	
	loss_weights={'value_head'+str(self.version_number): 0.5, 'policy_head'+str(self.version_number): 0.5}	
	)

```

## Why Deep Networks
Hypothetically, having a model that is one hidden layer would be able be able to produce similar results to any Deep Networks, so why do it?

There are multiple reasons for this structure. First, while it is mathmatically possible to emulate a deep model with a wide model. By creating a wide models it ends up emulating the function rather than actually being the function.

By creating a function with multiple layers with nonlinearities on each function, it allows the predictions to use relatively little weights, compared to the wide model. Therefore it creates a smoother model, and it actually learns the task at hand.

For example, at the bottom layer of a model, what you are concerened with is the movement of the pawns.	And perhaps, at the higher level, you would be worrying about how the different pieces protect each other. I raise these examples because it shows how you start from a local task, and it goes up the layers and addresses the total problem.

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
	#normalize batches to allow the gradients to converge faster
	x = LeakyReLU()(x)
	#Activation Function that is needed in Deep Networks to be effecive

	return (x)
```

However, with deeper models, there is a new problem introduced: Gradient vanishing. This concept happens because the gradients have to travel a large distance. So much of the original meaning of the gradient is lost. To combat this we introduce a layer called a resiudal layer. By adding the input of the current layer to the output, we are able to keep the original meaning of the gradients throughout the layers, because it is added directly to the output of the layer.


```python
x = self.conv_layer(input_block, filters, kernel_size)	
#Have the previous convolution layer
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
#adds the input to the output to preserve gradients
x = LeakyReLU()(x)
#basically another convolution layer
return (x)
```

##Value Head and Policy Head

```python
	def value_head(self, x):

		x = Conv2D(
		filters = 1
		, kernel_size = (1,1)
		, data_format="channels_first"
		, padding = 'same'
		, use_bias=False
		, activation='linear'
		, kernel_regularizer = regularizers.l2(self.reg_const)
		)(x)


		x = BatchNormalization(axis=1)(x)
		x = LeakyReLU()(x)

		x = Flatten()(x)
		#Flatten inputs
		x = Dense(
			20
			, use_bias=False
			, activation='linear'
			, kernel_regularizer=regularizers.l2(self.reg_const)
			)(x)

		x = LeakyReLU()(x)

		x = Dense(
			1
			, use_bias=False
			, activation='tanh'
			, kernel_regularizer=regularizers.l2(self.reg_const)
			, name = 'value_head'+str(self.version_number)
			)(x)

		#Use two dense layers to turn the shape into [b,1]


		return (x)

	def policy_head(self, x):

		x = Conv2D(
		filters = 2
		, kernel_size = (1,1)
		, data_format="channels_first"
		, padding = 'same'
		, use_bias=False
		, activation='linear'
		, kernel_regularizer = regularizers.l2(self.reg_const)
		)(x)

		x = BatchNormalization(axis=1)(x)
		x = LeakyReLU()(x)

		x = Flatten()(x)

		x = Dense(
			self.output_dim
			, use_bias=False
			, activation='linear'
			, kernel_regularizer=regularizers.l2(self.reg_const)
			, name = 'policy_head'+str(self.version_number)
			)(x)
		#Convert shape to outputdim so that they can be used as logits [b,output_dim]
		return (x)
```
For both the value_head function and policy_head function we start off with a Convolution layer with kernels of size 1. Then, we use Dense layers (xW+b) to shape the layers into the desired shapes.

[next part (tfjs)](tictac.md)