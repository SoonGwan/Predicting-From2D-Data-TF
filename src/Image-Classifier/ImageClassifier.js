/* eslint-disable jsx-a11y/alt-text */
import React from 'react';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as posenet from '@tensorflow-models/posenet';
import * as tf from '@tensorflow/tfjs';

const ImageClassifier = () => {
  let net;
  const webcamElement = document.getElementById('webcam');

  const app = async () => {
    console.log('Loading mobilenet..');

    net = await mobilenet.load();
    console.log('Successfully loaded model');

    const webcam = await tf.data.webcam(webcamElement, {
      resizeWidth: 224,
      resizeHeight: 224,
    });
    while (true) {
      const img = await webcam.capture();
      const result = await net.classify(img);

      document.getElementById(
        'console'
      ).innerHTML = `prediction : ${result[0].className} \n probability: ${result[0].probability}
      `;
      img.dispose();

      await tf.nextFrame();
    }
  };

  app();

  return (
    <div>
      <div id="console"></div>
      {/* <!-- Add an image that we will use to test --> */}
      <video autoplay playsinline muted id="webcam" width="224" height="224" />
    </div>
  );
};

export default ImageClassifier;
