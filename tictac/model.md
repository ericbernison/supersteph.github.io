# AlphaZero and Keras Models

## MCTS

### Game Trees
To understand MCTS, we first need an understanding of Game Trees.
Let's take a simple example of Tic Tac Toe. In this game, (and most other games) there are 2 main concepts: states and moves.
In Tic Tac Toe a state would be a board and any configuration of X's and O's in it with each configuration being a different state. However, not every configuration is going to be a state, as only states that are possible to reach following the rules of the game are considered valid game states.

A move is the way that you would play the game. Of course in tic tac toe not every single move is possible. For example, if a square already has an x placed on it you wouldn't be able to place another x on it. Another characteristic of a move is that it has the ability to move you to another state. (In some games, doing nothing could be a move) In Tic Tac Toe, however, making a move would guarantee that you get to another state
![alt text](https://github.com/supersteph.github.io/tictac/emptystate.gif "empty state")
(This is an game state where there are 9 possible moves next turn)

A game tree would be composed of states and moves. Where states are the nodes of the tree and moves are the edges along the tree.
In tic tac toe, the starting node would always be an empty 3x3 grid. From the starting node, there are 9 possible moves that are going to lead to a new state.

![alt text](https://github.com/supersteph.github.io/tictac/tree.png "game tree")

Notice on layer 1 where the states just consist of a X in one position. These are all going to be leaf nodes of the root node since you can get to these states from the root node. This is not going to be the full tree, however you can imagine that there are nine leaf nodes on the layer below the root node.

### MCTS in the Wild
The first thing that we want to understand is that Monte Carlo Tree Search (MCTS) can be applied to zero-sum games. A zero-sum game is a game where you win by having the opponent lose. It is convention to assign a value to each player in the game state. Where when a player wins they get the value of 1, when they lose they would get the value of -1, and when they tie both players get a value of 0. That way no matter the outcome, the sum of you and your opponents values are going to be equal to zero. To make algorithims to win at this game we go under the asssumption that your opponent is trying to make you lose (so that they can win), which leads to several algorithims.
Alpha Zero is based on the Monte Carlo Tree Search (MCTS) algorithm: the objective of this algorithim is to find the optimal (or very close to) action without exploring every node in the game tree. In games like tic tac toe -- where the game space is small -- it is very inexpensive to scour the entire game tree and apply an algorithim such as [minimax](https://www.baeldung.com/java-minimax-algorithm)
However, in games like chess, it is simply impossible to explore all the nodes in the tree. That is why we use the MCTS to get an approximation of the best move. The MCTS algorithm consists of four parts: Selection, Expansion, Simulation, Backpropagation. I am going to go into this in a relatively high-level, but if you want to read more into it there is a good blog [here](https://medium.com/@quasimik/monte-carlo-tree-search-applied-to-letterpress-34f41c86e238)

The first part is going to be selection. Assuming you have a tree in place, however the tree isn't completely explored. Also each node in the tree is going to have additional information other than the state, the amount of times visiting that node and the amount of wins from that node. The selection function is going to use an algorithim that balances exploring new stuff and the value of the state. This process happens in the middle of the tree; it is run repeadetly until it reaches an edge node.
![alt text](https://github.com/supersteph.github.io/tictac/selection.jpeg "selection")
Then the next part is expanding. You choose a random leaf node from the node that you selected and add it to your tree.

![alt text](https://github.com/supersteph.github.io/tictac/expansion.jpeg "expansion")

Then you simulate. In conventional MCTS you randomly move until you reach an ending state. You keep track of the wins and losses.
![alt text](https://github.com/supersteph.github.io/tictac/simulation.jpeg "simulation")
Then you backpropogate. You go up the tree that you have in place and depending on whether you won or lost you update the values in each node accordingly, in hopes for a better understanding of how that node is.
![alt text](https://github.com/supersteph.github.io/tictac/backprop.jpeg "backprop")
### In the Model
While our MCTS algorithim shares many similarities to the conventional MCTS, it also has many differences. The main one being that instead of simulating randomly, we basically trust our model to handle simulating. And, in the selection process we incorporate our model predictions in the function behind choosing the path to the leaf node. We also add every node in the expansion, instead of only doing one.

Our selection process is very similar

```python
while not currentNode.isLeaf():

	maxQU = -99999

	if currentNode == self.root:
		epsilon = config.EPSILON
		nu = np.random.dirichlet([config.ALPHA] * len(currentNode.edges))
	else:
		epsilon = 0
		nu = [0] * len(currentNode.edges)
	#Set up the values to be used in the function for selecting

	Nb = 0
	for action, edge in currentNode.edges:
		Nb = Nb + edge.stats['N']
	#sum up the total edges visited in the leaf nodes

	for idx, (action, edge) in enumerate(currentNode.edges):

		U = self.cpuct * \
			((1-epsilon) * edge.stats['P'] + epsilon * nu[idx] )  * \
			np.sqrt(Nb) / (1 + edge.stats['N'])
		#a function of P (from the model prediction) and N the number of times visiting this node representing how unknown going down the current edge is 
		Q = edge.stats['Q']
		# The average value of going down this path representing how likely you are going to win

		if Q + U > maxQU:
			# Get the max Q+U and keep track of the action and edge that it is
			maxQU = Q + U
			simulationAction = action
			simulationEdge = edge

	newState, value, done = currentNode.state.takeAction(simulationAction) 
	#take the action with the highes q+u
	currentNode = simulationEdge.outNode
	#Get the node that the edge leads to
	breadcrumbs.append(simulationEdge)
	#add the edge to breadcrumbs so that you know what to backfill

#Ends once we get to a leaf node
```
This code will go through a path by starting at the root node and going down the game tree one level at a time. At each level, you have a `currentNode` and going through all the edges from that node. Then, you find the edge with the maximum Q+U and then set `currentNode` as the node that that paticular edge ends at.
In our function, Q (the average value) is going to represent how likely you are to win if you traverse down that edge. U is going to represent how unknown that edge is. You want to go down edges that you know are going to lead to a win, and you also want to go down edges that have a lot of potential and that is why you have a Q and a U. Note that the U is trying to find nodes that the model deems to have a high reward, and it hasn't explored that much.


```python
if done == True:

	value, probs, allowedActions = self.get_preds(leaf.state)
	#get_preds is the function that plugs the state of the game into the model
	probs = probs[allowedActions]
	#Only keep the valid moves in the probs array

	for idx, action in enumerate(allowedActions):
		newState, _, _ = leaf.state.takeAction(action)
		#Get the state if you take the action
		if newState.id not in self.mcts.tree:
			node = mc.Node(newState)
			self.mcts.addNode(node)
			#If the state doesn't exist in the node add the new node to the tree
		else:
			node = self.mcts.tree[newState.id]
			#Otherwise just get the information from the node from the tree

		newEdge = mc.Edge(leaf, node, probs[idx], action)
		leaf.edges.append((action, newEdge))
		#Connect the new node to the current leaf node
		
else:
	#This means that the value is already set, which means you do not need to evaluate or add the node

return ((value, breadcrumbs))

```
This is going to be the expansion part of MCTS. Which is pretty much the exact same as the textbook MCTS defintion.

```python
for edge in breadcrumbs:
	#go up the tree
	playerTurn = edge.playerTurn
	if playerTurn == currentPlayer:
		direction = 1
	else:
		direction = -1
	# get whether value should be positive or negative
	edge.stats['N'] = edge.stats['N'] + 1
	edge.stats['W'] = edge.stats['W'] + value * direction
	edge.stats['Q'] = edge.stats['W'] / edge.stats['N']
	#update stats

```
Pretty standard backfill code. Go up the edges of the tree and update them as you go along.

[next part (model)](supersteph.github.io/tictac/modelagain.md)