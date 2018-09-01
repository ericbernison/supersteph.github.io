
# Basics of Tensorflow JS

## Introduction
I had a keras model trained so that it could play tic tac toe. However, I wanted to be able to have an online demo so I could show people that I knew tensorflow. So I followed the tensorflow guide and ran into a whole lot of errors.

## Getting model.json
The first thing you would do in download the tensorflow js converter in pip.
```
pip install tensorflowjs
```
note you have to have a tensorflow version of 1.9 or higher for this to work.

Then with the h5 model in hand you execute this statement
```
tensorflowjs_converter --input_format keras model.h5 target_dir
```
If this runs successfully you should have a model.json file and a bunch of shards or something.
## TensorflowJS in Javascript
Perhaps one of the most intersting things about tfjs is that when trying to use the model the tf.loadModel is set up as an callback function. If you don't want your entire script to be asynchronous then you can use an async function to set up the model like I did	
There are two basic ways to set up your javascript file for tfjs.
1. In the corresponding html file you add this statement
...```
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"> </script>
```
2. In the begenning of the Javascript file add an import statement
...```
import * as tf from '@tensorflow/tfjs';
```
I would not recommend the second version because this would require you to set up tensorflow js in your server enviorment instead of just getting it from the web during run time which in my opinion is a easier solution. Of course, if you are crunched on time during run time you may want to use the second option

Another thing to note about the tensors is the shape that they are expecting
```
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

I had an error where tfjs complained about an error where they wanted a shape of [,2,3,3] and I gave them shape [2,3,3] this one infuriated me, but I added an extra diemension to my original array and it seemed to work.

It went along the lines of this 
```
tfjs@latest:2 Uncaught (in promise) Error: 2 of 68 weights are not set: value_head/kernel,policy_head/kernel.
```
In the model.json file the kernel weights were set as value_head_1 and policy_head_1. This problem originates from having two variables with the same name. There are two ways to solve this, to manually go into the model.json file and change the value_head_1 to value_head or give the two variables two different names value_head1 and value_head0 which is what I ended up doing

## Conclusion
While still irritiating and painful to deal with, TensorflowJS has potential.