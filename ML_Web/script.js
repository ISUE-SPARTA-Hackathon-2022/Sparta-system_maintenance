// Grab a reference to our status text element on the web page.
// Initially we print out the loaded version of TFJS.
const status = document.getElementById('status');
status.innerText = 'Loaded TensorFlow.js - version: ' + tf.version.tfjs;

// Specify location of our Model.json file we uploaded to the Glitch.com CDN.
const MODEL_URL = './JS/model.json';
// Specify a test value we wish to use in our prediction.
// Here we use 950, so we expect the result to be close to 950,000.
const TEST_VALUE = 950.0

// Create an asynchronous function.
async function run() {
    // Load the model from the CDN.
    const model = await tf.loadLayersModel(MODEL_URL);

    // Print out the architecture of the loaded model.
    // This is useful to see that it matches what we built in Python.
    console.log(model.summary());

    // Create a 1 dimensional tensor with our test value.
    const input = tf.tensor1d([TEST_VALUE]);

    // Actually make the prediction.
    const result = model.predict(input);

    // Grab the result of prediction using dataSync method
    // which ensures we do this synchronously.
    status.innerText = 'Input of ' + TEST_VALUE + 
        'sqft predicted as $' + result.dataSync()[0];
}

// Call our function to start the prediction!
run();