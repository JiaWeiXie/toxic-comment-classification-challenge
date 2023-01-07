# Toxic Comment Classification Challenge

[kaggle dataset](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)

## System dependency

- make
- python 3.9

```bash
$ nvcc -V
$ nvidia-smi

$ sudo apt-get install -y nvidia-docker2
$ sudo apt-get install -y libcudnn8 libcudnn8-dev
```

## Installtion

```bash
$ make venv
$ source .venv/bin/activate
$ make install mode=dev
```

## Run notebook

```bash
$ make notebook
```

## Run server

```bash
$ make runserver
```