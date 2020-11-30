/* eslint-disable jsx-a11y/alt-text */
import React from 'react';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as posenet from '@tensorflow-models/posenet';
import * as tf from '@tensorflow/tfjs';

const ImageClassifier = () => {
  let net;
  const app = async () => {
    console.log('Loading mobilenet..');

    net = await mobilenet.load();
    console.log('Successfully loaded model');

    const imgEl = document.getElementById('img');
    const result = await net.classify(imgEl);
    console.log(result);
  };

  app();

  return (
    <div>
      <div id="console"></div>
      {/* <!-- Add an image that we will use to test --> */}
      <img
        id="img"
        crossOrigin="anonymous"
        src="https://i.imgur.com/JlUvsxa.jpg"
        width="227"
        height="227"
      />
    </div>
  );
};

export default ImageClassifier;
