import React from 'react';
import TensorPredict from './TensorPredict/TensorPredict';
import HandWriteCNN from './CNN/HandwriteCNN';
import PosturalEmotion from './Postural-Emotion/PosturalEmotion';
import ImageClassifier from './Image-Classifier/ImageClassifier';

const App = () => {
  return (
    <div>
      {/* <TensorPredict /> */}
      {/* <HandWriteCNN /> */}
      {/* <PosturalEmotion /> */}
      <ImageClassifier />
    </div>
  );
};

export default App;
