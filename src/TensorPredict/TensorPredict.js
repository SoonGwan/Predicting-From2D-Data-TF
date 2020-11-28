import React from 'react';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
// import { tfvis, scatterplot } from '@tensorflow/tfjs-vis';
const TensorPredict = () => {
  const getData = async () => {
    const carsDataResponse = await fetch(
      'https://storage.googleapis.com/tfjs-tutorials/carsData.json'
    );
    const carsData = await carsDataResponse.json();
    console.log(carsData.length);
    const cleaned = carsData
      .map((car) => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower,
      }))
      .filter((car) => car.mpg != null && car.horsepower != null);
    return cleaned;
  };

  const createModel = () => {
    const model = tf.sequential();

    //싱글 인풋 레이어
    model.add(
      tf.layers.dense({
        inputShape: [1],
        units: 50,
        useBias: true,
        activation: 'sigmoid',
      })
    );
    // 이렇게하면 네트워크에 입력 레이어 가 추가되며 , 이는 dense하나의 숨겨진 유닛이 있는 레이어에 자동으로 연결됩니다 . dense층은 매트릭스 (호출 곱하여 그 입력 그런 층의 일종 가중치 후) 및 (착신 번호 추가 바이어스 결과를 참조). 이것이 네트워크의 첫 번째 계층이므로 inputShape. 는 inputShape것입니다 [1]우리가 가지고 있기 때문에 1우리의 입력 (해당 자동차의 마력)로 수입니다.
    // units레이어에서 가중치 행렬의 크기를 설정합니다. 여기서 1로 설정하면 데이터의 각 입력 특성에 대해 1 개의 가중치가 있음을 의미합니다.

    model.add(tf.layers.dense({ units: 1 }));
    // model.add(tf.layers.dense({ units: 50, activation: 'sigmoid' }));
    // 위의 코드는 출력 레이어를 생성합니다. 숫자 를 출력하고 싶기 때문에 설정 units했습니다 .11

    return model;
  };

  const convertToTensor = (data) => {
    return tf.tidy(() => {
      // 데이터 섞기
      tf.util.shuffle(data);

      const inputs = data.map((d) => d.horsepower);
      const labels = data.map((d) => d.mpg);

      const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();

      const normalizedInputs = inputTensor
        .sub(inputMin)
        .div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor
        .sub(labelMin)
        .div(labelMax.sub(labelMin));

      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        inputMax,
        inputMin,
        labelMax,
        labelMin,
      };
    });
  };
  // convertToTensor();

  const trainModel = async (model, inputs, labels) => {
    model.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError,
      metics: ['mse'],
    });

    const batchSize = 32;
    const epochs = 50;

    return await model.fit(inputs, labels, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks(
        { name: 'Training Performance' },
        ['loss', 'mse'],
        { height: 200, callbacks: ['onEpochEnd'] }
      ),
    });
  };

  const testModel = (model, inputData, normalizationData) => {
    const { inputMax, inputMin, labelMax, labelMin } = normalizationData;

    const [xs, preds] = tf.tidy(() => {
      const xs = tf.linspace(0, 1, 100);
      const preds = model.predict(xs.reshape([100, 1]));
      // 모델에 대한 100개의 새로운 예제를 생성하는 것 [num_examples, num_features_per_example]훈련을 할 때 와 비슷한 모양 ( ) 을 가져야합니다 .

      const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
      const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

      return [unNormXs.dataSync(), unNormPreds.dataSync()];
      // .dataSync() typedarray 텐서에 저장된 값을 가져오는데 사용할 수 있는 방법
    });

    const predictedPoints = Array.from(xs).map((val, i) => {
      return { x: val, y: preds[i] };
    });
    const originalPoints = inputData.map((d) => ({
      x: d.horsepower,
      y: d.mpg,
    }));

    tfvis.render.scatterplot(
      { name: 'Model Predictions vs Original Data' },
      {
        values: [originalPoints, predictedPoints],
        series: ['original', 'predicted'],
      },
      {
        xLabel: 'Horsepower',
        yLabel: 'MGH',
        height: 300,
      }
    );
  };

  const run = async () => {
    const model = createModel();
    tfvis.show.modelSummary({ name: 'Model Summary' }, model);
    const data = await getData();
    const values = data.map((d) => ({
      x: d.horsepower,
      y: d.mpg,
    }));

    const tensorData = convertToTensor(data);
    const { inputs, labels } = tensorData;
    await trainModel(model, inputs, labels);
    console.log('The end Train');

    testModel(model, data, tensorData);
    tfvis.render.scatterplot(
      { name: 'Horsepower v MPG' },
      { values },
      {
        xLabel: 'Horsepower',
        yLabel: 'MPG',
        height: 300,
      }
    );
  };
  run();

  return (
    <>
      <div></div>
    </>
  );
};

export default TensorPredict;
