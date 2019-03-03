import * as tf from '@tensorflow/tfjs';
import { SSL_OP_ALL } from 'constants';
const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;
const predictionsElement = document.getElementById('predictions');
let output = document.getElementById("output");
let link = document.getElementById("followLink");
const maizeClasses = ['TLB',
  'Aphid',
  'Armyworm',
  'banded leaf',
  'Common rust',
  'CutWorm',
  'Hairy Caterpillar',
  'Normal',
  'Stalk Rot',
  'White Grub'];
const cottonPestClasses = ['Leaf Hoppers Cotton ',
  'Mealybug Cotton',
  'Mirids Cotton',
  'Pink Bollworm',
  'Thrips Cotton '];
const cottonDiseaseClasses = ['Cotton Leaf Curl Virus (CLCuD)',
  'Bacterial Blight',
  'Cotton Plant',
  'Grey Mildew',
  'Root Rot Cotton']
const ricePestClasses = ['Rice_Brown_Stem_Hopper',
  'Rice_Gundhi_Bug',
  'Rice_Leaf_Folder',
  'Rice_Normal',
  'Rice_Yellow_Stem_Borer']
let btn = document.getElementById("ch");
btn.onclick = (e) => {
  var checkedValue = document.querySelector('.choice:checked').value;
  localStorage.setItem("choice", checkedValue);
  predict(window.imag, localStorage.getItem('choice'));
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

const MAIZENET_MODEL_PATH =
  // tslint:disable-next-line:max-line-length
  'https://raw.githubusercontent.com/shikhar-scs/sssss/master/maize_model/model.json';

const COTTONDNET_MODEL_PATH =
  'https://raw.githubusercontent.com/pratyush1687/sss2/master/model.json'

const COTTONPNET_MODEL_PATH =
  "https://raw.githubusercontent.com/shikhar-scs/sssss/master/cotton_model/model.json"
const RICENET_MODEL_PATH =
"https://raw.githubusercontent.com/pratyush1687/ssss/master/model.json"

const IMAGE_SIZE = 224;

let maizenet;
let cottondnet;
let cottonpnet;
let ricenet;
const netDemo = async () => {
  status('Loading model...');

  maizenet = await tf.loadLayersModel(MAIZENET_MODEL_PATH);
  cottondnet = await tf.loadLayersModel(COTTONDNET_MODEL_PATH);
  cottonpnet = await tf.loadLayersModel(COTTONPNET_MODEL_PATH);
  ricenet = await tf.loadLayersModel(RICENET_MODEL_PATH);



  maizenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
  cottondnet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
  cottonpnet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
  ricenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
  status('waiting for upload...');

  document.getElementById('file-container').style.display = '';

};
netDemo();
/**
 * Given an image element, makes a prediction through mobilenet returning the
 * probabilities of the top K classes.
 */
async function predict(imgElement, choice) {
  status('Predicting...');

  const startTime = performance.now();

  let finalModel; let classes
  if (choice == 'Maize') {
    finalModel = maizenet;
    classes = maizeClasses;
  }
  else if (choice == 'Cotton-pest') {
    // console.info("calall");
    finalModel = cottonpnet;
    classes = cottonPestClasses;
  }
  else if (choice == 'Rice') {
    finalModel = ricenet;
    classes = ricePestClasses;
  }
  else if (choice == 'Cotton-Disease') {
    finalModel = cottondnet;
    classes = cottonDiseaseClasses;
  }
  const logits = tf.tidy(() => {
    // tf.browser.fromPixels() returns a Tensor from an image element.
    const img = tf.browser.fromPixels(imgElement).toFloat();

    const offset = tf.scalar(127.5);
    // Normalize the image from [0, 255] to [-1, 1].
    const normalized = img.sub(offset).div(offset);

    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    // Make a prediction through mobilenet.
    return finalModel.predict(batched);
  });
  const values = await logits.data();
  console.info(values)
  // sort(values)
  let pred = classes[parseInt(indexOfMax(values))];
  let result = [];
  for (let i = 0; i < classes.length; i++) {
    // let obj = {};
    result.push({
      class: classes[i],
      probability: values[i]
    })
  }
  result.sort(function (a, b) {
    return b.probability-a.probability;
  })
  let s= "<h1>Results</h1><br>"
  result = result.slice(0,3);
  result.forEach(x=>{
    s+=`${x.class} : ${Math.round(x.probability*10000)/100}%
        <i><a href=${"http://localhost:3000/mlResult?q=" + pred}>click here to know measures</a><i><br>`
  })
  
  output.innerHTML = `${s}`;
  // link.href = "http://localhost:3000/mlResult?q=" + pred;
  // link.innerHTML = "click here";

  const totalTime = performance.now() - startTime;
  status(`Done in ${Math.floor(totalTime)}ms`);

}

const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    const idx = i;
    // Closure to capture the file information.
    reader.onload = e => {
      // Fill the image & call predict.
      let img = document.createElement('img');
      img.src = e.target.result;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
      let choice = localStorage.getItem("choice");
      img.onload = () => window.imag = img;
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
    status('waiting to predict...');
  }
});
