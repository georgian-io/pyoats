<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Apache 2.0 License][license-shield]][license-url]
[![Last Commit][last_commit-shield]][last_commit-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/georgian-io/oats">
    <img src="https://github.com/georgian-io/pyoats/raw/main/static/oats.png" alt="Logo" width="auto" height="80">
  </a>

<h3 align="center"> OATS</h3>

  <p align="center">
    Quick and Easy Outlier Detection for Time Series 
    <br />
    <a href="https://georgian-io.github.io/pyoats-docs/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/georgian-io/oats">View Demo</a>
    ·
    <a href="https://github.com/georgian-io/pyoats/issues">Report Bug</a>
    ·
    <a href="https://github.com/georgian-io/pyoats/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#models">Models</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
Adapting existing outlier detection & prediction methods into a **time series outlier detection** system is not a simple task. Good news: **OATS** has done the heavy lifting for you! 

We present a straight-forward interface for popular, state-of-the-art detection methods to assist you in your experiments. In addition to the models, we also present different options when it comes to selecting a final threshold for predictions.

**OATS** seamlessly supports both univariate and multivariate time series regardless of the model choice and guarantees the same output shape, enabling a modular approach to time series anoamly detection.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With
[![Python][Python.org]][Python-url] [![Poetry][Python-Poetry.org]][Poetry-url]

[![Pytorch][Pytorch.org]][Torch-url]  [![PytorchLightning][PytorchLightning.ai]][Lightning-url] [![TensorFlow][TensorFlow.org]][TF-url] [![Numpy][Numpy.org]][Numpy-url]

[![Darts][Darts]][Darts-url] [![PyOD][PyOD]][PyOD-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started
<br />
<div align="center">
    <img src="https://github.com/georgian-io/pyoats/raw/main/static/example-sine_wave.png" alt="Usage Example" width="600" height="auto">
  </a>
  </div>


### Prerequisites
[![Python][Python.org]][Python-url] >= 3.8

For Docker Install:
 [![Docker][Docker.com]][Docker-url]

For Local Install:
 [![Poetry][Python-Poetry.org]][Poetry-url]


### Installation
#### PyPI
1. Install package via pip
   ```sh
   pip install pyoats
   ```
  
#### Docker
1. Clone the repo
    ```sh
    git clone https://github.com/georgian-io/pyoats.git && cd pyoats 
    ```
2. Build image
    ```sh
    docker build -t pyoats . 
    ```
3. Run Container
    ```sh 
    # CPU Only
    docker run -it pyoats
    
    # with GPU
    docker run -it --gpus all pyoats
    ```
    
#### Local
1. Clone the repo
    ```sh
    git clone https://github.com/georgian-io/pyoats.git && cd pyoats 
    ```
2. Install via Poetry
    ```sh 
    poetry install
    ```



<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Getting Anomaly Score
```python 
from oats.models import NHiTSModel

model = NHiTSModel(window=20, use_gpu=True)
model.fit(train)
scores = model.get_scores(test)
```
### Getting Threshold
```python 
from oats.threshold import QuantileThreshold

t = QuantileThreshold()
threshold = t.get_threshold(scores, 0.99)
anom = scores > threshold
```
_For more examples, please refer to the [Documentation](https://georgian-io.github.io/pyoats-docs/)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Models -->
## Models


_For more details about the individual models, please refer to the [Documentation](https://georgian-io.github.io/pyoats-docs/)_

Model | Type | Multivariate Support* | Requires Fitting | DL Framework Dependency | Paper | Reference Model
--- | :---: | :---: | :---: | :---: | :---: |  :---: 
`ARIMA` | Predictive | ⚠️ | ✅ |  | | [`statsmodels.ARIMA`](https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima.model.ARIMA.html)
`FluxEV` | Predictive | ⚠️ | ✅ |  | [📝](https://dl.acm.org/doi/10.1145/3437963.3441823) | 
`LightGBM` | Predictive | ⚠️ | ✅ |  |  | [`darts.LightGBM`](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.gradient_boosted_model.html)
`Moving Average` | Predictive | ⚠️ |  |  |  | 
`N-BEATS` | Predictive | ✅ | ✅ | [![Pytorch][Pytorch.org]][Torch-url] | [📝](https://openreview.net/forum?id=r1ecqn4YwB) | [`darts.NBEATS`](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nbeats.html)
`N-HiTS` | Predictive | ✅ | ✅ | [![Pytorch][Pytorch.org]][Torch-url] | [📝](https://arxiv.org/abs/2201.12886) | [`darts.NHiTS`](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nhits.html)
`RandomForest` | Predictive | ⚠️ | ✅ | | | [`darts.RandomForest`](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.random_forest.html)
`Regression` | Predictive | ⚠️ | ✅ | | | [`darts.Regression`](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.regression_model.html)
`RNN` | Predictive | ✅ | ✅ | [![Pytorch][Pytorch.org]][Torch-url] | | [`darts.RNN`](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.rnn_model.html)
`Temporal Convolution Network` | Predictive | ✅ | ✅ | [![Pytorch][Pytorch.org]][Torch-url] | [📝](https://arxiv.org/abs/1803.01271) | [`darts.TCN`](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tcn_model.html)
`Temporal Fusion Transformers` | Predictive | ✅ | ✅ | [![Pytorch][Pytorch.org]][Torch-url] | [📝](https://arxiv.org/abs/1912.09363) | [`darts.TFT`](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tft_model.html)
`Transformer` | Predictive | ✅ | ✅ | [![Pytorch][Pytorch.org]][Torch-url] | [📝](https://arxiv.org/abs/1706.03762) | [`darts.Transformer`](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.transformer_model.html)
`Isolation Forest` | Distance-Based | ✅ | ✅ | || [`pyod.IForest`](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.iforest)
`Matrix Profile` | Distance-Based | ✅ |  | | [📝](https://www.cs.ucr.edu/~eamonn/MatrixProfile.html) | [`stumpy`](https://github.com/TDAmeritrade/stumpy)
`TranAD` | Reconstruction-Based | ✅ | ✅ | [![TensorFlow][TensorFlow.org]][Torch-url] | [📝](https://arxiv.org/abs/2201.07284) | [`tranad`](https://github.com/imperial-qore/TranAD)
`Variational Autoencoder` | Reconstruction-Based | ✅ | ✅ | [![TensorFlow][TensorFlow.org]][Torch-url] |   [📝](https://arxiv.org/abs/1312.6114) | [`pyod.VAE`](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.vae)
`Quantile` | Rule-Based | ⚠️ |  |  || 





**\*** For models with ⚠️, score calculation is done separately along each column. This implicitly assumes independence of covariates, which means that **the resultant anomaly scores do not take into account of inter-variable dependency structures.**

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- ROADMAP -->
## Roadmap

- [ ] Automatic hyper-parameter tuning
- [ ] More examples 
- [ ] More preprocessors
- [ ] More models from `pyod`

See the [open issues](https://github.com/georgian-io/pyoats/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/amazing_feature`)
3. Commit your Changes (`git commit -m 'Add some amazing_feature'`)
4. Push to the Branch (`git push origin feature/amazing_feature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

<div align="left">
  <a href="https://www.georgian.io">
    <img src="https://s34372.pcdn.co/wp-content/uploads/2022/03/Georgian_Blue.png" alt="Logo" width="auto" height="80">
  </a>
  </div>


|<!-- -->|<!-- -->|<!-- -->|<!-- -->|
|---|---|---|---|
| __Benjamin Ye__ | [![Github][BenGithub]][BenLinkedIn-url] | [![LinkedIn][BenLinkedIn]][BenLinkedIn-url] |  [![eMail][eMail]][BenEmail-url] 


Project Link: [https://github.com/georgian-io/oats](https://github.com/georgian-io/oats)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
I would like to thank my colleagues from Georgian for all the help and advice provided along the way.
* [Angeline Yasodhara](mailto:angeline@georgian.io)
* [Akshay Budhkar](mailto:akshay@georgian.io)
* [Borna Almasi](mailto:borna@georgian.io)
* [Parinaz Sobhani](mailto:parinaz@georgian.io)
* [Rodrigo Ceballos Lentini](mailto:rodrigo@georgian.io)

I'd also like to extend my gratitude to all the contributors at [`Darts`][Darts-url] (for time series predictions) and [`PyOD`][PyOD-url] (for general outlier detection), whose projects have enabled a straight-forward extension into the domain of time series anomaly detection.

Finally, it'll be remiss of me to not mention [DATA Lab @ Rice University](https://cs.rice.edu/~xh37/index.html), whose wonderful [`TODS`][TODS-url] package served as a major inspiration for this project. Please check them out especially if you're looking for AutoML support.

[![Darts][Darts]][Darts-url] [![PyOD][PyOD]][PyOD-url] [![TODS][TODS]][TODS-url]

 
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/georgian-io/oats.svg?style=for-the-badge
[contributors-url]: https://github.com/georgian-io/pyoats/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/georgian-io/contributors.svg?style=for-the-badge
[forks-url]: https://github.com/georgian-io/pyoats/network/members
[stars-shield]: https://img.shields.io/github/stars/georgian-io/oats.svg?style=for-the-badge
[stars-url]: https://github.com/georgian-io/pyoats/stargazers
[issues-shield]: https://img.shields.io/github/issues/georgian-io/oats.svg?style=for-the-badge
[issues-url]: https://github.com/georgian-io/pyoats/issues
[license-shield]: https://img.shields.io/github/license/georgian-io/oats.svg?style=for-the-badge
[license-url]: https://github.com/georgian-io/pyoats/blob/master/LICENSE
[last_commit-shield]: https://img.shields.io/github/last-commit/georgian-io/oats.svg?style=for-the-badge
[last_commit-url]: https://github.com/georgian-io/pyoats/commits/



<!-- Deps Links -->
[Python-Poetry.org]: https://img.shields.io/badge/Poetry-60A5FA?style=for-the-badge&logo=poetry&logoColor=white
[Poetry-url]: https://www.python-poetry.org/
[Python.org]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[PyTorch.org]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[Torch-url]: https://pytorch.org/
[TensorFlow.org]: https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white
[TF-url]: https://www.tensorflow.org/
[Numpy.org]: https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white
[Numpy-url]: https://www.numpy.org/
[Darts]: https://img.shields.io/badge/Repo-Darts-2100FF?style=for-the-badge&logo=github&logoColor=white
[Darts-url]: https://github.com/unit8co/darts
[PyOD]: https://img.shields.io/badge/Repo-PyOD-000000?style=for-the-badge&logo=github&logoColor=white
[PyOD-url]: https://github.com/yzhao062/pyod
[TODS]: https://img.shields.io/badge/Repo-TODS-29B48C?style=for-the-badge&logo=github&logoColor=white
[TODS-url]: https://github.com/datamllab/tods
[Docker.com]: https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white
[Docker-url]: https://docker.com
[PyTorchLightning.ai]: https://img.shields.io/badge/lightning-792EE5?style=for-the-badge&logo=pytorchlightning&logoColor=white
[Lightning-url]: https://www.pytorchlightning.ai/



[BenLinkedIn]: https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white
[BenLinkedIn-url]: https://www.linkedin.com/in/benjaminye/

[eMail]: https://img.shields.io/badge/EMail-EA4335?style=for-the-badge&logo=gmail&logoColor=white
[BenEmail-url]: mailto:benjamin.ye@georgian.io

[BenGithub]: https://img.shields.io/badge/Profile-14334A?style=for-the-badge&logo=github&logoColor=white
[BenGithub-url]: https://www.linkedin.com/in/benjaminye/
