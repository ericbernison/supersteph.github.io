

console.log("hi yinger");

var model;  
start();
async function start() {
    //arabic or english
    
    var testarray = [[[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]]]];

    var test = tf.tensor(testarray);

    //const testagain = test.reshape([,2,3,3]);
    //load the model 
    model = await tf.loadModel('model/model.json');
    
    //warm up 
    const output = model.predict(test);
    
    logits_array = output[1].dataSync();
    console.log(logits_array);
}


var x = "/tictac/krestik.gif";
var o = "/tictac/nolik.gif";  
var blank = "/tictac/blank.jpg";
var pause = 0;
var all = 0;
var a = 0;
var b = 0;
var c = 0;
var d = 0;
var e = 0;
var f = 0;
var g = 0;
var h = 0;
var i = 0;  
var temp="";
var ok = 0;
var cf = 0;
var choice=9;
var aRandomNumber = 0;
var comp = 0; 
var t = 0;
var wn = 0;
var ls = 0;
var ts = 0;

function convert(){
  var testarray = [[[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]]]
  if(a!=0){
    if(a==1){
      testarray[0][1][0][0]=1;
    }
    else if(a==2){
      testarray[0][0][0][0]=1;
    }
  }
  if(b!=0){
    if(b==1){
      testarray[0][1][0][1]=1;
    }
    else if(b==2){
      testarray[0][0][0][1]=1;
    }
  }
  if(c!=0){
    if(c==1){
      testarray[0][1][0][2]=1;
    }
    else if(c==2){
      testarray[0][0][0][2]=1;
    }
  }
  if(d!=0){
    if(d==1){
      testarray[0][1][1][0]=1;
    }
    else if(d==2){
      testarray[0][0][1][0]=1;
    }
  }
  if(e!=0){
    if(e==1){
      testarray[0][1][1][1]=1;
    }
    else if(e==2){
      testarray[0][0][1][1]=1;
    }
  }
  if(f!=0){
    if(f==1){
      testarray[0][1][1][2]=1;
    }
    else if(a==2){
      testarray[0][0][1][2]=1;
    }
  }
  if(g!=0){
    if(g==1){
      testarray[0][1][2][0]=1;
    }
    else if(g==2){
      testarray[0][0][2][0]=1;
    }
  }
  if(h!=0){
    if(h==1){
      testarray[0][1][2][1]=1;
    }
    else if(h==2){
      testarray[0][0][2][1]=1;
    }
  }
  if(i!=0){
    if(i==1){
      testarray[0][1][2][2]=1;
    }
    if(i==2){
      testarray[0][0][2][2]=1;
    }
  }
  return testarray;
}
function logicOne() {
  if ((a==1)&&(b==1)&&(c==1)) all=1;
  if ((a==1)&&(d==1)&&(g==1)) all=1;
  if ((a==1)&&(e==1)&&(i==1)) all=1;
  if ((b==1)&&(e==1)&&(h==1)) all=1;
  if ((d==1)&&(e==1)&&(f==1)) all=1;
  if ((g==1)&&(h==1)&&(i==1)) all=1;
  if ((c==1)&&(f==1)&&(i==1)) all=1;
  if ((g==1)&&(e==1)&&(c==1)) all=1;
  if ((a==2)&&(b==2)&&(c==2)) all=2;
  if ((a==2)&&(d==2)&&(g==2)) all=2;
  if ((a==2)&&(e==2)&&(i==2)) all=2;
  if ((b==2)&&(e==2)&&(h==2)) all=2;
  if ((d==2)&&(e==2)&&(f==2)) all=2;
  if ((g==2)&&(h==2)&&(i==2)) all=2;
  if ((c==2)&&(f==2)&&(i==2)) all=2;
  if ((g==2)&&(e==2)&&(c==2)) all=2;
  if ((a != 0)&&(b != 0)&&(c != 0)&&(d != 0)&&(e != 0)&&(f != 0)&&(g != 0)&&(h != 0)&&(i != 0)&&(all == 0)) all = 3;
} 
function clearOut() {
  document.game.you.value="0";
  document.game.computer.value="0";
  document.game.ties.value="0";
}
function checkSpace() {
  if ((temp=="A")&&(a==0)) {
    ok=1;
    if (cf==0) a=1;
    if (cf==1) a=2;
  }
  if ((temp=="B")&&(b==0)) {
    ok=1;
    if (cf==0) b=1;
    if (cf==1) b=2;
  }
  if ((temp=="C")&&(c==0)) {
    ok=1;
    if (cf==0) c=1;
    if (cf==1) c=2;
  }
  if ((temp=="D")&&(d==0)) {
    ok=1;
    if (cf==0) d=1;
    if (cf==1) d=2;
  }
  if ((temp=="E")&&(e==0)) {
    ok=1;
    if (cf==0) e=1;
    if (cf==1) e=2;
  }
  if ((temp=="F")&&(f==0)) {
    ok=1
    if (cf==0) f=1;
    if (cf==1) f=2;
  }
  if ((temp=="G")&&(g==0)) {
    ok=1
    if (cf==0) g=1;
    if (cf==1) g=2;
  }
  if ((temp=="H")&&(h==0)) {
    ok=1;
    if (cf==0) h=1;
    if (cf==1) h=2;
  }
  if ((temp=="I")&&(i==0)) {
    ok=1;
    if (cf==0) i=1; 
    if (cf==1) i=2; 
  }
}
function yourChoice(chName) {
  pause = 0;
  if (all!=0) ended();
  if (all==0) {
    cf = 0;
    ok = 0;
    temp=chName;
    checkSpace();
    if (ok==1) {
      document.images[chName].src = x;
    }
    if (ok==0)taken();
    process();
    if ((all==0)&&(pause==0)) myChoice();
  }
}
function taken() {
  alert("This cell in not empty! Try another")
  pause=1;
}

function indexOfMax(arr) {
    if (arr.length === 0) {
        return -1;
    }

    var max = arr[0];
    var maxIndex = 0;

    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }

    return maxIndex;
}
function setnegative(logits){
  if(a!=0){
    logits[0] = -100
  }
  if(b!=0){
    logits[1] = -100
  }
  if(c!=0){
    logits[2] = -100
  }
  if(d!=0){
    logits[3] = -100
  }
  if(e!=0){
    logits[4] = -100
  }
  if(f!=0){
    logits[5] = -100
  }
  if(g!=0){
    logits[6] = -100
  }
  if(h!=0){
    logits[7] = -100
  }
  if(g!=0){
    logits[8] = -100
  }

  var outputs = [];
  var sum = 0;
  for(var i = 0; i<9;i++){
    const temporary = Math.pow(Math.E,logits);
    outputs.push(temporary);
    sum = sum + temporary;
  }

  for(var i = 0; i<9;i++){
    outputs[i]=outputs[i]/sum;
  }

  return outputs;


}
function ichoose(){
  const testarray = convert();
  console.log(testarray);
  var test = tf.tensor(testarray);
  const output = model.predict(test);
  const logits = output[1].dataSync();
  const newlogits = setnegative(logits);

  const choice = indexOfMax(logits);
  console.log(choice);

  if(choice==0){
    temp="A";
  }
  if(choice==1){
    temp="B";
  }
  if(choice==2){
    temp="C";
  }
  if(choice==3){
    temp="D";
  }
  if(choice==4){
    temp="E";
  }
  if(choice==5){
    temp="F";
  }
  if(choice==6){
    temp="G";
  }
  if(choice==7){
    temp="H";
  }
  if(choice==8){
    temp="I";
  }
}
function myChoice() {
  temp="";
  ok = 0;
  cf=1;
  ichoose();
  checkSpace();
  document.images[temp].src= o;
  process();
}
function ended() {
  alert("Game over! To play once more press a button 'New Game'")
}
function process() {
  logicOne();
  if (all==1){ alert("You win!"); wn++; }
  if (all==2){ alert("You lose!"); ls++; }
  if (all==3){ alert("Draw!"); ts++; }
  if (all!=0) {
    document.game.you.value = wn;
    document.game.computer.value = ls;
    document.game.ties.value = ts;
  }
}
function playAgain() {
  if (all==0) {
    if(confirm("Âû óâåðåíû ?")) reset();
  }
  if (all>0) reset();
}
function reset() {
  all = 0;
  a = 0;
  b = 0;
  c = 0;
  d = 0;
  e = 0;
  f = 0;
  g = 0;
  h = 0;
  i = 0;
  temp="";
  ok = 0;
  cf = 0;
  choice=9;
  aRandomNumber = 0;
  comp = 0;
  document.images.A.src= blank;
  document.images.B.src= blank;
  document.images.C.src= blank;
  document.images.D.src= blank;
  document.images.E.src= blank;
  document.images.F.src= blank;
  document.images.G.src= blank;
  document.images.H.src= blank;
  document.images.I.src= blank;
  if (t==0) { t=2; myChoice(); }
  t--;
}
var ie4 = (document.all) ? true : false;
var nn4 = (document.layers) ? true : false;

document.getElementById("A").onclick = function() {yourChoice('A')};
document.getElementById("B").onclick = function() {yourChoice('B')};
document.getElementById("C").onclick = function() {yourChoice('C')};
document.getElementById("D").onclick = function() {yourChoice('D')};
document.getElementById("E").onclick = function() {yourChoice('E')};
document.getElementById("F").onclick = function() {yourChoice('F')};
document.getElementById("G").onclick = function() {yourChoice('G')};
document.getElementById("H").onclick = function() {yourChoice('H')};
document.getElementById("I").onclick = function() {yourChoice('I')};
document.getElementById("button").onclick = function() {playAgain()};
