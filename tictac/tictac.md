
# Basics of Tensorflow JS

## Introduction
I had a keras model trained so that it could play tic tac toe. However, I wanted to be able to have an online demo so I could show people that I knew tensorflow. At this moment, there isn't a lot of ways to convert a model so that it can be used javascript. I decided to try and use tensorflowjs because it seemed like the most accessible way. So I followed the tensorflowjs guide and ran into a whole lot of errors.

## Getting model.json
The first thing you would do is download the tensorflow js converter in pip.
```
pip install tensorflowjs
```
(Note: you have to have a tensorflow version of 1.9 or higher for this to work.)

I am going on the assumption that you already have a trained keras model, I used the code from AppliedDataSciencePartners and modified the code so that the model is learning tictactoe. With a Keras model you run this function
```python
model.save(run_folder+name+ '.h5')
```

With an h5 file in hand, you can run the following function.
```
tensorflowjs_converter --input_format keras model.h5 target_dir
```
If this runs successfully you should have a model.json file and a bunch of shards in the target_dir. The target_dir is going to make up the model.

## TensorflowJS in Javascript	
There are two basic ways to set up your javascript file for tfjs.
1. In the html file where you reference your own js file you add this additional statement
```
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"> </script>
```
2. In the begenning of the Javascript file add an import statement
```javascript
import * as tf from '@tensorflow/tfjs';
```
I would not recommend the second version because this would require you to set up tensorflow js in your server enviorment instead of just getting it from the web during run time which in my opinion is a easier solution. Of course, if you are crunched on time during run time you may want to use the second option.

In fact, it seems a lot more prone to error. In chrome when I run the second version I get an error where the compiler doesn't recognize the "*" symbol.

Another thing to note about the tensors is the shape that they are expecting

I had an error where tfjs complained about an error where they wanted a shape of [,2,3,3] and I gave them shape [2,3,3] but I added an extra diemension to my original array and it seemed to work. However, reshaping it directly to [,2,3,3] doesn't work. It needs to be in the shape of [1,2,3,3]

Perhaps one of the most intersting things about tfjs is that when trying to use the model the tf.loadModel is set up as an callback function. If you don't want your entire script to be asynchronous then you can use an async function to set up the model like I did and then call from a synchrounous.
(If you want to know about async functions check [this](https://medium.com/codebuddies/getting-to-know-asynchronous-javascript-callbacks-promises-and-async-await-17e0673281ee) out)
```python
async function start() {
    //arabic or english
    
    var testarray = [[[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]]]];

    var test = tf.tensor(testarray);

    //const testagain = test.reshape([,2,3,3]);
    //load the model 
    model = await tf.loadModel('tfjs/model.json');
    
    //warm up 
    const output = model.predict(test);
    
    logits_array = output[1].dataSync();
    console.log(logits_array);
}
```
The predict function is pretty self-explanitory except that you need to feed it an input in the format of tf.tensor for it to be able to work.

the await keyword means that the function means that you are going to finish the execution of tf.loadModel() before you continue to the next line of code. While this is very self-explanitory in synchronous code async functions are another subject that I'm not going to go into.

My model returns a tuple of [predictedValue, logits] each of them are in the form of a tensor. I did not use the value so I got the logits tensor and performed dataSync on it which made the output turn from tf.tensor form into a javascript array where I would be able to manipulate it easily.

It went along the lines of this 
```
tfjs@latest:2 Uncaught (in promise) Error: 2 of 68 weights are not set: value_head/kernel,policy_head/kernel.
```
In the model.json file the kernel weights were set as value_head_1 and policy_head_1. This problem originates from having two variables with the same name. There are two ways to solve this, to manually go into the model.json file and change the value_head_1 to value_head or give the two variables two different names value_head1 and value_head0 which is what I ended up doing.

This seems to be an error on tensorflowjs's part. Because while importing the model from an h5 file works perfectly fine, when making multiple of the same variable and then converting in a form that tfjs can use results in the kernel file adding an automatic ```_x``` with x being the number. This would work, however it fails to update the other parts that is looking for the kernel variable.

## Conclusion
While still irritiating and painful to deal with, TensorflowJS has potential.

# Demo
<html>
<head>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"> </script>


<script src="backend.js" defer></script> 
<img border="0" height="0"
src="krestik.gif" width="0">
<img border="0" height="0"
src="nolik.gif" width="0">
</head>
<body>
	
<form name="game">
<div align="center"><center><table border="0">
<TBODY>
<tr>
<td><table border="1" borderColor="#000000" cellPadding="0" cellSpacing="0">
<TBODY>
<tr>
<td><a id ="A"><img border="0" height="61" name="A"
src="blank.jpg" width="56"></a></td>
<td><a id ="B"><img border="0" height="61" name="B"
src="blank.jpg" width="56"></a></td>
<td><a id = "C"><img border="0" height="61" name="C"
src="blank.jpg" width="56"></a></td>
</tr>
<tr>
<td><a id="D"><img border="0" height="61" name="D"
src="blank.jpg" width="56"></a></td>
<td><a id = "E"><img border="0" height="61" name="E"
src="blank.jpg" width="56"></a></td>
<td><a id="F"><img border="0" height="61" name="F"
src="blank.jpg" width="56"></a></td>
</tr>
<tr>
<td><a id="G"><img border="0" height="61" name="G"
src="blank.jpg" width="56"></a></td>
<td><a id="H"><img border="0" height="61" name="H"
src="blank.jpg" width="56"></a></td>
<td><a id = "I"><img border="0" height="61" name="I"
src="blank.jpg" width="56"></a></td>
</tr>
</TBODY>
</table>
</td>
<td><table>
<TBODY>
<tr colspan="2">
<td><font face="MS Sans Serif" size="1"><b>Score:</b></font></td>
</tr>
<tr>
<td><font face="MS Sans Serif" size="1"><input name="you" size="5"
style="font-family: MS Sans Serif; font-size: 1"></font></td>
<td><font face="MS Sans Serif" size="1">You</font></td>
</tr>
<tr>
<td><font face="MS Sans Serif" size="1"><input name="computer" size="5"
style="font-family: MS Sans Serif; font-size: 1"></font></td>
<td><font face="MS Sans Serif" size="1">Computer</font></td>
</tr>
<tr>
<td><font face="MS Sans Serif" size="1"><input name="ties" size="5"
style="font-family: MS Sans Serif; font-size: 1"></font></td>
<td><font face="MS Sans Serif" size="1">Draw</font></td>
</tr>
</TBODY>
</table>
</td>
</tr>
</TBODY>
</table>
</center></div><div align="center"><center><p><input id ="button"
value="New Game"
style="font-family: MS Sans Serif; font-size: 1; font-weight: bold"> </p>
</center></div>
</form> 

</body>
</html>