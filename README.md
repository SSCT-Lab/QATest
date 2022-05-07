# QATest: A Uniform Fuzzing Framework for Question Answering Systems

## Environment

- python=3.6
- Pytorch=1.8.0

- GPU is needed.

You can create the environment through [Conda](https://docs.conda.io/en/latest/):

```shell
conda create -n gentle python=3.6
conda activate gentle
pip install -r requirements.txt
```

## Generating tests with QATest

```sh
python main.py --dataset boolq --system unifiedqa --strategy qatest
```

