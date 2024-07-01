## Code for Machine Translation of Transformer
The codes are based on [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py). 


### Usage

#### IWSLT 2014

1. Download and tokenize the dataset:

```bash
cd data/iwslt14/
bash prepare-iwslt14.sh
```

2. Preprocess the dataset:

```bash
cd ../../
bash preprocess.iwslt14.sh
```

3. Train the model:

- Fully connected network:

```bash
bash iwslt_FC.sh
```

- CHTs:

```bash
bash iwslt_CHTs.sh
```

- SET:

```bash
bash iwslt_SET.sh
```

4. Evaluate the model:

```bash
bash eval_iwslt.sh ${beam_size} ${model_path}
```

#### Multi-30k

Download and preprocess Multi-30k:

```bash
python preprocess.py -train_src data/Multi30k/train.en -train_tgt data/Multi30k/train.de -valid_src data/Multi30k/val.en -valid_tgt data/Multi30k/val.de -save_data data/Multi30k/processed.noshare -src_seq_length 256 -tgt_seq_length 256 -src_vocab_size 40000 -tgt_vocab_size 40000
```

#### WMT17

Download and preprocess WMT17:

```bash
cd data/wmt17/
bash prepare-wmt14.sh
cd ../../
bash preprocess.wmt17.sh
```

