# Cannistraci-Hebb Training (CHT) and Cannistraci-Hebb Training soft rule (CHTs)

## Epitopological Learning and Cannistraci-Hebb Network Shape Intelligence Brain-Inspired Theory for Ultra-Sparse Advantage in Deep Learning

Yingtao Zhang<sup>1,2,3</sup>, Jialin Zhao<sup>1,2,3</sup>, Wenjing Wu<sup>1,2,3</sup>, Alessandro Muscoloni<sup>1,2,4</sup> & Carlo Vittorio Cannistraci<sup>1,2,3,4</sup>

<sup>1</sup> Center for Complex Network Intelligence (CCNI)  
<sup>2</sup> Tsinghua Laboratory of Brain and Intelligence (THBI)  
<sup>3</sup> Department of Computer Science  
<sup>4</sup> Department of Biomedical Engineering  
Tsinghua University, Beijing, China

## Brain-inspired Sparse Training in MLP and Transformers with Network Science Modeling via Cannistraci-Hebb Soft Rule

Yingtao Zhang<sup>1,2,4</sup>, Jialin Zhao<sup>1,2,4</sup>, Ziheng Liao<sup>1,2,4</sup>, Wenjing Wu<sup>1,2,4</sup>, Umberto Michieli<sup>5</sup> & Carlo Vittorio Cannistraci<sup>1,2,3,4</sup>

<sup>1</sup> Center for Complex Network Intelligence (CCNI), Tsinghua Laboratory of Brain and Intelligence (THBI)  
<sup>2</sup> Department of Computer Science  
<sup>3</sup> Department of Biomedical Engineering  
<sup>4</sup> Tsinghua University, Beijing, China  
<sup>5</sup> University of Padova, Italy

## Setup

1. Create a new conda environment:

```markdown
conda create -n chts python=3.10
conda activate chts
```

2. Install relevant packages:

```bash
pip install -r requirements.txt
```

3. Compile the python-c code:

```bash
python setup.py build_ext --inplace
```

## Usage

### MLP

Navigate to the MLP directory:

```bash
cd mlp_and_cnn
```

#### CHTs on EMNIST-MLP task

```bash
python run.py --batch_size 32 --dataset EMNIST --network_structure mlp --weight_decay 5e-04 --regrow_method CH2_L3n_soft --init_mode swi --linearlr --epochs 100 --learning_rate 0.025 --cuda_device 0 --dim 2 --update_interval 1 --self_correlated_sparse --no_log --chain_removal --zeta 0.3 --remove_method ri --seed 0 --sparsity 0.99 --T_decay linear --dst_scheduler --adaptive_zeta
```

#### CHTs on EMNIST-MLP task + EM_S

```bash
python run.py --batch_size 32 --dataset EMNIST --network_structure mlp --weight_decay 5e-04 --regrow_method CH2_L3n_soft --init_mode swi --linearlr --epochs 100 --learning_rate 0.025 --cuda_device 0 --dim 2 --update_interval 1 --self_correlated_sparse --no_log --chain_removal --zeta 0.3 --remove_method ri --seed 0 --sparsity 0.99 --T_decay linear --dst_scheduler --EM_S
```

Note:

- `--remove_method` can be chosen from weight_magnitude, weight_magnitude_soft, ri, ri_soft 
- `--self_correlated_sparse` means using Correlated Sparse Topological initialization
- For Bipartite Small World (BSW) model, activate `--WS --beta $YOUR_BETA_VALUE`
- For Bipartite Scale-Free (BSF) model, activate `--BA`

### Transformer

Navigate to the Transformer directory:

```bash
cd Transformer
```

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

## Citation

If you use our code, please consider citing:

### CHT

```
@inproceedings{
zhang2024epitopological,
title={Epitopological learning and Cannistraci-Hebb network shape intelligence brain-inspired theory for ultra-sparse advantage in deep learning},
author={Yingtao Zhang and Jialin Zhao and Wenjing Wu and Alessandro Muscoloni and Carlo Vittorio Cannistraci},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=iayEcORsGd}
}
```

### CHTs

```
@article{202406.1136,
	doi = {10.20944/preprints202406.1136.v1},
	url = {https://doi.org/10.20944/preprints202406.1136.v1},
	year = 2024,
	month = {June},
	publisher = {Preprints},
	author = {Yingtao Zhang and Jialin Zhao and Ziheng Liao and Wenjing Wu and Umberto Michieli and Carlo Vittorio Cannistraci},
	title = {Brain-Inspired Sparse Training in MLP and Transformers with Network Science Modeling via Cannistraci-Hebb Soft Rule},
	journal = {Preprints}
}
```

