# DeepAL: Deep Active Learning in Python

Python implementations of the following active learning algorithms:

- Random Sampling
- Least Confidence [1]
- Margin Sampling [2]
- Entropy Sampling [3]
- Uncertainty Sampling with Dropout Estimation [4]
- Bayesian Active Learning Disagreement [4]
- Cluster-Based Selection [5]
- Adversarial margin [6]

## Prerequisites 

- numpy            1.21.2
- scipy            1.7.1
- pytorch          1.10.0
- torchvision      0.11.1
- scikit-learn     1.0.1
- tqdm             4.62.3
- ipdb             0.13.9

You can also use the following command to install conda environment

```
conda env create -f environment.yml
```

## Demo 

```
  python demo.py \
      --n_round 10 \
      --n_query 1000 \
      --n_init_labeled 10000 \
      --dataset_name MNIST \
      --strategy_name RandomSampling \
      --seed 1
```

Please refer [here](https://arxiv.org/abs/2111.15258) for more details.

## Citing

If you use our code in your research or applications, please consider citing our paper.

```
@article{Huang2021deepal,
    author    = {Kuan-Hao Huang},
    title     = {DeepAL: Deep Active Learning in Python},
    journal   = {arXiv preprint arXiv:2111.15258},
    year      = {2021},
}
```

## Reference

[1] A Sequential Algorithm for Training Text Classifiers, SIGIR, 1994

[2] Active Hidden Markov Models for Information Extraction, IDA, 2001

[3] Active learning literature survey. University of Wisconsin-Madison Department of Computer Sciences, 2009

[4] Deep Bayesian Active Learning with Image Data, ICML, 2017

[5] Active Learning for Convolutional Neural Networks: A Core-Set Approach, ICLR, 2018

[6] Adversarial Active Learning for Deep Networks: a Margin Based Approach, arXiv, 2018

## Docker

- `docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --build-arg WANDB_API_KEY=<API_KEY> --force-rm -t deepal .`
- `docker run --rm --shm-size="2g" -v ${PWD}:/app -w /app -p 8888:8888 -p 8049:8049 --name deepal -itd deepal bash`
- `NV_GPU=5 nvidia-docker run -it --rm --shm-size=20g --ulimit memlock=-1 -v ${PWD}:/app -w /app --name deepal -itd deepal bash`
- `docker exec -it deepal bash`

### Run jupyter notebooks
- `jupyter lab --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token='deepal' &`
- Acessar: `http://localhost:8888/lab?token=deepal`

### Exportar env
- `conda env export --from-history > environment.yml`
