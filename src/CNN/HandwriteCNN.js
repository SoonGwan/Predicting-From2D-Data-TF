import React from 'react';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { MnistData } from './data';

const HandWriteCNN = () => {
  const showExamples = async (data) => {
    console.log(data);
    const surface = tfvis
      .visor()
      .surface({ name: 'INPUT DATA EXAM', tab: 'INPUTDATA' });

    // 예제 가지고 오기
    const examples = data.nextTestBatch(20);
    const numExamples = examples.xs.shape[0];

    // 캔버스 엘레먼트 랜더 examp
    for (let i = 0; i < numExamples; i++) {
      console.log(numExamples);
      const imageTensor = tf.tidy(() => {
        // 28 x 28 사이즈
        return examples.xs
          .slice([i, 0], [1, examples.xs.shape[1]])
          .reshape([28, 28, 1]);
      });

      const canvas = document.createElement('canvas');
      canvas.width = 28;
      canvas.height = 28;
      canvas.style = 'margin 4px;';
      await tf.browser.toPixels(imageTensor, canvas);
      surface.drawArea.appendChild(canvas);

      imageTensor.dispose();
    }
  };

  const getModel = () => {
    const model = tf.sequential();

    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;

    model.add(
      tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
      })
    );

    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    model.add(
      tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
      })
    );
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    model.add(tf.layers.flatten());

    const NUM_OUTPUT_CLASSES = 10;
    model.add(
      tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax',
      })
    );

    const optimizer = tf.train.adam();
    model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });

    return model;
  };

  const train = async (model, data) => {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
      name: 'Model Training',
      tab: 'Model',
      styles: { height: '1000px' },
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE = 1000;

    const [trainXs, trainYs] = tf.tidy(() => {
      const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
      return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
    });

    const [testXs, testYs] = tf.tidy(() => {
      const d = data.nextTestBatch(TEST_DATA_SIZE);
      return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
    });

    return model.fit(trainXs, trainYs, {
      batchSize: BATCH_SIZE,
      validationData: [testXs, testYs],
      epochs: 10,
      shuffle: true,
      callbacks: fitCallbacks,
    });
  };

  const run = async () => {
    const data = new MnistData();
    await data.load();
    await showExamples(data);
    const model = getModel();
    tfvis.show.modelSummary(
      { name: 'Model Architecture', tab: 'Model' },
      model
    );
    await train(model, data);
  };

  run();
  return (
    <>
      <div></div>
    </>
  );
};

export default HandWriteCNN;
