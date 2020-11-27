import React from 'react';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
// import { tfvis, scatterplot } from '@tensorflow/tfjs-vis';
const App = () => {
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
    model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));

    // 아웃풋 레이어 추가
    model.add(tf.layers.dense({ units: 1 }));

    return model;
  };

  const run = async () => {
    const model = createModel();
    tfvis.show.modelSummary({ name: 'Model Summary' }, model);
    const data = await getData();
    const values = data.map((d) => ({
      x: d.horsepower,
      y: d.mpg,
    }));

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

export default App;
