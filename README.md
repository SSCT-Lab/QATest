# QATest: A Uniform Fuzzing Framework for Question Answering Systems

## Environment

- python=3.6
- Pytorch=1.8.0

- GPU is needed.

You can create the environment through [Conda](https://docs.conda.io/en/latest/):

```shell
conda create -n qatest python=3.6
conda activate gentle
pip install -r requirements.txt
```


## Generating tests with QATest

```sh
# Data preprocessing, get 500 test cases for generation:
python preprocess.py --dataset squad --system unifiedqa

# generating tests:
python main.py --dataset squad --system unifiedqa --strategy qatest
```

