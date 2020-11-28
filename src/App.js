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
    // 이렇게하면 네트워크에 입력 레이어 가 추가되며 , 이는 dense하나의 숨겨진 유닛이 있는 레이어에 자동으로 연결됩니다 . dense층은 매트릭스 (호출 곱하여 그 입력 그런 층의 일종 가중치 후) 및 (착신 번호 추가 바이어스 결과를 참조). 이것이 네트워크의 첫 번째 계층이므로 inputShape. 는 inputShape것입니다 [1]우리가 가지고 있기 때문에 1우리의 입력 (해당 자동차의 마력)로 수입니다.
    // units레이어에서 가중치 행렬의 크기를 설정합니다. 여기서 1로 설정하면 데이터의 각 입력 특성에 대해 1 개의 가중치가 있음을 의미합니다.

    model.add(tf.layers.dense({ units: 1 }));
    // 위의 코드는 출력 레이어를 생성합니다. 숫자 를 출력하고 싶기 때문에 설정 units했습니다 .11

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
