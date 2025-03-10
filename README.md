This is a repository of the following papers:
 - [Peer-to-peer Deep Learning With Non-IID Data](https://www.sciencedirect.com/science/article/abs/pii/S0957417422021777)
 - [Multi-task peer-to-peer learning using the BERT transformer model](https://www.sciencedirect.com/science/article/abs/pii/S0167739X23004053)
 - [Evaluacija generalizacije znanja decentraliziranih agenata u okruženju heterogenih skupova podataka](https://) -> **Under Review at MIPRO conference**
 - [An Overview of Autonomous Connection Establishment Methods in Peer-to-Peer Deep Learning](https://) -> **Under Review**
 - [Sequence-to-sequence models in peer-to-peer learning: A practical application](https://) -> **Under Review**

---


# Sequence-to-sequence models in peer-to-peer learning: A practical application

The code can be run through [notebooks](/notebooks) with names starting with **ASR-**.
To run the experiments please download `UserLibri` dataset from [https://www.kaggle.com/datasets/google/userlibri](https://www.kaggle.com/datasets/google/userlibri).
The `UserLibri` dataset should be extracted and placed in `data/userlibri` resulting in the following hierarchy:
```ssh
project_root/
│
├── data/
│   ├── userlibri/
│   │   ├── UserLibri
│   │   │   ├── audio_data
│   │   │   ├── lm_data
```

The `LJ Speech` dataset will be automatically downloaded when running the code for the first time.  


<br>
<br>

# An Overview of Autonomous Connection Establishment Methods in Peer-to-Peer Deep Learning

The code can be run through [notebooks](/notebooks) with names starting with **Peer-**.
Before running, prepared datasets need to be downloaded from [google drive](https://drive.google.com/drive/folders/1p1RqD0eeTMxXgyFB7WxVgSXEmkHMqPdV?usp=share_link).
The datasets should be unzipped in the following directories:
- Reddit (grouped by reddit topic) (clients_category.zip) at directory `data/reddit/`

All result files of the experiments can be found on a shared [google drive](https://drive.google.com/drive/folders/1wu21lUgfCDK8_h8YoJAevGZjn14kyWLV?usp=sharing) and can also be reproduced following instructions bellow. To visualise existing results, use [`plot/conn_viz.py`](plot/conn_viz.py) and place the downloaded data (conns.zip) in `log/` directory.

<br>
<br>

# Evaluacija generalizacije znanja decentraliziranih agenata u okruženju heterogenih skupova podataka

The code can be run through a [notebook](/notebooks) with the name **Robustness**.

All result files of the experiments can be found on a shared [google drive](https://drive.google.com/drive/folders/1wu21lUgfCDK8_h8YoJAevGZjn14kyWLV?usp=sharing) and can also be reproduced following instructions bellow. To visualise existing results, use [`plot/robust_viz.py`](plot/robust_viz.py) and place the downloaded data (rob.zip) in `log/` directory.


<br>
<br>

# Multi-task peer-to-peer learning using the BERT transformer model

This paper presents and evaluates a novel approach that utilizes the popular BERT transformer model to enable collaboration between agents learning two different learning tasks: next-word prediction and named-entity recognition. The evaluation of the studied approach revealed that collaboration among agents, even when working towards separate objectives, can result in mutual benefits, mainly when the communication between agents is carefully considered. The multi-task collaboration led to a statistically significant increase of 11.6% in the mean relative accuracy compared to the baseline results for individual tasks.
The code can be run through [notebooks](/notebooks) with names starting with **BERT-**.
Before running, prepared datasets need to be downloaded from [google drive](https://drive.google.com/drive/folders/1p1RqD0eeTMxXgyFB7WxVgSXEmkHMqPdV?usp=share_link).
The datasets should be unzipped in the following directories:
- CoNNL-2003 (conll.zip) at directory `data/ner/conll/`
- Few-NERD (few_nerd.zip) at directory `data/ner/few_nerd/`
- Reddit (bert-reddit.zip) at directory `data/reddit/bert_clients`
- StackOverflow (bert-stackoverflow.zip) at directory `data/stackoverflow/bert_clients`

All result files of the experiments can be found on a shared [google drive](https://drive.google.com/drive/folders/1wu21lUgfCDK8_h8YoJAevGZjn14kyWLV?usp=sharing) and can also be reproduced following instructions bellow. To visualise existing results, use [`plot/mt_viz.py`](plot/mt_viz.py) and place the downloaded data (mt.zip) in `log/` directory.

<br>
<br>


# Peer-to-peer Deep Learning With Non-IID Data (P2P-BN)

This paper proposes using Batch Normalization (BN) layers as an aid in normalizing non-IID data across decentralized agents. A variant of early stopping technique is developed that, in combination with BN layers, acts as a tool for fine-tuning the agent’s local model. Other decentralized algorithms used in the paper experiments are also a part of this project. 

---

Experiments were performed on a cleaned [Reddit](https://github.com/TalwalkarLab/leaf) dataset as it presents as a viable non-IID dataset.


Our experiments showed that using BN layers in a NN model benefits all decentralized algorithms. Figure below shows that all algorithms show positive feedback when using the model containing BN layers in a directed and undirected ring communication scheme.


![exp_1](imgs/exp_1.svg)



P2P-BN substantially outperforms other decentralized algorithms when trained in a sparse topology with three neighbors per agent.

![exp_2](imgs/exp_2.svg)




This good performance is due to using a variant of early stopping by only training BN layers when an agent cannot further improve the model. We show that in the figure below.

![exp_3](imgs/exp_es.svg)



All result files of the experiments can be found on a shared [google drive](https://drive.google.com/drive/folders/1wu21lUgfCDK8_h8YoJAevGZjn14kyWLV?usp=sharing) and can also be reproduced following instructions bellow. To visualise existing results, use [`plot/experiment_viz.py`](plot/experiment_viz.py) and place the downloaded (results.zip) data in `log/` directory.

## Running simulations

#### Dataset preparation

Preprocessed dataset can be downloaded from a shared [google drive](https://drive.google.com/drive/folders/1p1RqD0eeTMxXgyFB7WxVgSXEmkHMqPdV?usp=sharing). Unpack the zip into the `data` directory. When completed, folder `data/reddit/reddit/clients/` must contain all `.h5` files containing the training, validation and test data. Tokenizer must be placed in `data/reddit/reddit/`

Alternatively, you can recreate this dataset by following the info in [Reddit.md](data/reddit/Reddit.md).


**Simulations can be run either through [notebooks](/notebooks) or through cmd.**

### Run through cmd

#### Running agent simulation
```
$ test_p2p.py -h
usage: test_p2p.py [-h] --agent AGENT --clients CLIENTS [--dataset DATASET]
                   [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--seed SEED]
                   [--model_v MODEL_V] [--lr LR] [--agent_pars AGENT_PARS]
                   [--graph_type GRAPH_TYPE] [--neighbors NEIGHBORS]
                   [--directed] [--vary VARY]

optional arguments:
  -h, --help              show this help message and exit
  --agent AGENT           Agent class to be used in simulations
  --clients CLIENTS       Number of clients to be used in simulation
  --dataset DATASET       Dataset to use in simulations
  --batch_size BATCH_SIZE Batch size (default 50)
  --epochs EPOCHS         Number of cumulative epoch to train (default 30)
  --seed SEED             Seed (default None)
  --model_v MODEL_V       Model version to be used in simulation (default 4)
  --lr LR                 Agent learning rate (default 0.005)
  --agent_pars AGENT_PARS Json-type string with custom agent parameters (default None)
  --graph_type GRAPH_TYPE Graph type to create as a communication base (default sparse)
  --neighbors NEIGHBORS   Number of neighbors each agent has (default 3)
  --directed              Set this flag for directed communication (default false)
  --vary VARY             Time-varying interval of changing communication matrix (default -1)
```

Example:

```
$ python test_p2p.py --agent P2PAgent --clients 100 --epochs 100 --dataset reddit
```

Parameter `agent_pars` expects a dictionary of name-values forwarded to the agent class. Surround dictionary with single quotes and values within dict with double-quotes.
```
$ python test_p2p.py --agent P2PAgent --clients 100 --epochs 100 --agent_pars='{"early_stopping": true}'
```

#### Running Federated Learning simulation

```
$ test_fl.py -h
usage: test_fl.py [-h] --clients CLIENTS --sample_size SAMPLE_SIZE
                  [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--seed SEED]
                  [--model_v MODEL_V] [--slr SLR] [--clr CLR]

optional arguments:
  -h, --help                show this help message and exit
  --clients CLIENTS         Number of clients to be used in simulation
  --sample_size SAMPLE_SIZE Number of clients to sample in each FL round
  --epochs EPOCHS           Number of cumulative epoch to train (default 30)
  --batch_size BATCH_SIZE   Batch size (default 50)
  --seed SEED               Seed (default None)
  --model_v MODEL_V         Model version to be used in simulation (default 2)
  --slr SLR                 Server learning rate (default 0.005)
  --clr CLR                 Client learning rate (default 0.005)
```

Example:
```
$ python test_fl.py --clients 100 --sample_size 10
```
