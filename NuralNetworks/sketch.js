// Daniel Shiffman
// https://youtu.be/N3ZnNa01BPM



//              This is a nural network that predicts the XOR bitwise function 
//              mostly this is just a trivial example for learning tensorflow.js


let nn;
let model;

let resolution = 20;
let cols;
let rows;

let xs;

const train_xs = tf.tensor2d([ // Inputs
  [0, 0],
  [1, 0],
  [0, 1],
  [1, 1]
]);

const train_ys = tf.tensor2d([ // Outputs
  [0],
  [1],
  [1],
  [0]
]);

function setup() {
  createCanvas(400, 400);
  cols = width / resolution;
  rows = height / resolution;

  // Create the input data

  // These are the inputs they are using, they are arrays of 2 floats from 0-1
  // They don't need to do anything extra to the data

  let inputs = [];
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      let x1 = i / cols;
      let x2 = j / rows;
      inputs.push([x1, x2]);
    }
  }
  xs = tf.tensor2d(inputs);


  model = tf.sequential();

  // This is the only origional hidden layer
  let hiddenNodes = 4;
  let hidden = tf.layers.dense({ // it is dense, meaning that it is fully connected
    inputShape: [2],
    units: hiddenNodes,
    activation: 'sigmoid' // and the activation function is sigmoid
  });


  // I added this layer and played around with changing some the shape of both layers.
  // It was weird how depending on the number of nodes in the layers the time to reach 
  // a good result were completely different.
  // It worked best with only one layer, four nodes, and the adam optimizer.
  let hidden1 = tf.layers.dense({ 
    inputShape: [hiddenNodes],
    units: 2,
    activation: 'sigmoid'
  });

  // This is the output layer, they are single floats from 0-1
  let output = tf.layers.dense({
    units: 1,
    activation: 'sigmoid'
  });
  model.add(hidden);
  model.add(hidden1);
  model.add(output);

  const optimizer = tf.train.adam(0.2); // Adam is the best optimizer function, some of the others don't seem to do much of anything
  model.compile({
    optimizer: optimizer,
    loss: 'meanSquaredError' // The loss function is Mean Squared Error
  })

  setTimeout(train, 10);

}

function train() {
  trainModel().then(result => {
    setTimeout(train, 10);
  });
}

function trainModel() {
  return model.fit(train_xs, train_ys, {
    shuffle: true,
    epochs: 2
  });
}

function draw() {
  background(0);





  tf.tidy(() => {
    let ys = model.predict(xs);
    let y_values = ys.dataSync();

    // The purpose of this example was to be simple so the validation is visualized and easy to figure out
    // if the 0,0 and 1,1 corners are black (0.00), and the 1,0 and 0,1 corners are white (1.00) then it worked

    let index = 0;
    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < rows; j++) {
        let br = y_values[index] * 255
        fill(br);
        rect(i * resolution, j * resolution, resolution, resolution);
        fill(255 - br);
        textSize(8);
        textAlign(CENTER, CENTER);
        text(nf(y_values[index], 1, 2), i * resolution + resolution / 2, j * resolution + resolution / 2)
        index++;
      }
    }
  });

}



