const STATUS = document.getElementById('status');
const ML_CLASS = document.getElementById('ml_class');
const VIDEO = document.getElementById('webcam');

const MOBILE_NET_INPUT_WIDTH = 500;
const MOBILE_NET_INPUT_HEIGHT = 500;

const STOP_DATA_GATHER = -1;
const CLASS_NAMES = ["Export", "Salable", "Waste"];

let custom_model = undefined;
let gatherDataState = STOP_DATA_GATHER;
let videoPlaying = false;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];
let predict = false;

async function loadMobileNetFeatureModel() {
  console.log(window.location.href);
  
  custom_model = await tf.loadLayersModel('localstorage://app/static/');
  STATUS.innerText = 'by: system_maintenance';
  
  // Warm up the model by passing zeros through it once.
  tf.tidy(function () {
    let answer = custom_model.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
    console.log(answer.shape);
  });
}

async function predictLoop() {
  if (custom_model) {
    tf.tidy(function() {
      let videoFrameAsTensor = tf.browser.fromPixels(VIDEO).div(255);
      let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor,[MOBILE_NET_INPUT_HEIGHT, 
          MOBILE_NET_INPUT_WIDTH], true);

      let imageFeatures = mobilenet.predict(resizedTensorFrame.expandDims());
      let prediction = model.predict(imageFeatures).squeeze();
      let highestIndex = prediction.argMax().arraySync();
      let predictionArray = prediction.arraySync();

      ML_CLASS.innerText = CLASS_NAMES[highestIndex];
      // confidence Math.floor(predictionArray[highestIndex] * 100)
    });

    window.requestAnimationFrame(predictLoop);
  }
}

loadMobileNetFeatureModel();
predictLoop();