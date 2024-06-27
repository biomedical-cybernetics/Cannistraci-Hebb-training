## Cannistraci-Hebb Training (CHT) and Cannistraci-Hebb Training soft rule (CHTs)

--------

#### Epitopological Learning and Cannistraci-Hebb Network Shape Intelligence Brain-Inspired Theory for Ultra-Sparse Advantage in Deep Learning

Yingtao Zhang<sup>1,2,3</sup>, Jialin Zhao<sup>1,2,3</sup>, Wenjing Wu<sup>1,2,3</sup>, Alessandro Muscoloni<sup>1,2,4</sup>  
& Carlo Vittorio Cannistraci<sup>1,2,3,4</sup>  
<sup>1</sup> Center for Complex Network Intelligence (CCNI)  
<sup>2</sup> Tsinghua Laboratory of Brain and Intelligence (THBI)  
<sup>3</sup> Department of Computer Science  
<sup>4</sup> Department of Biomedical Engineering  
Tsinghua University, Beijing, China  


#### Brain-inspired Sparse Training in MLP and Transformers with Network Science Modeling via Cannistraci-Hebb Soft Rule


Yingtao Zhang<sup>1,2,4</sup>, Jialin Zhao<sup>1,2,4</sup>, Ziheng Liao<sup>1,2,4</sup>, Wenjing Wu<sup>1,2,4</sup>  
Umberto Michieli<sup>5</sup> & Carlo Vittorio Cannistraci<sup>1,2,3,4</sup>

<sup>1</sup> Center for Complex Network Intelligence (CCNI), Tsinghua Laboratory of Brain and Intelligence (THBI)
<sup>2</sup> Department of Computer Science
<sup>3</sup> Department of Biomedical Engineering
<sup>4</sup> Tsinghua University, Beijing, China
<sup>5</sup> University of Padova, Italy


#### Setup

------

Step 1: Create a new conda environment:

```
conda create -n cht python=3.10
conda activate cht
```



Step 2: Install relevant packages

```
pip3 install torch=1.31.1+cu117
pip install transformers=4.36.2 sentencepiece=0.1.99 datasets=2.16.1 bitsandbytes=0.42.0
pip install accelerate=0.26.1
pip install pybind11
```

Step 3: Compile the python-c code

```
python setup.py build_ext --inplace
```

#### Usage

----

CHT on MNIST-MLP task

```
python run.py \
	--learning_rate 0.01 --epochs 100 \
	--regrow_method CH3_L3 --init_mode swi --chain_removal --self_correlated_sparse \
	--dim 2 --bias --linearlr --early_stop --update_interval 3 --dataset MNIST \
```



SET on MNIST-MLP task

```
python run.py \
	--learning_rate 0.01 --epochs 100 \
	--regrow_method random --init_mode kaiming \
	--dim 2 --bias --linearlr --update_interval 3 --dataset MNIST \
```

-------

CHT on ResNet152-CIFAR100 task

```
python run.py \
	--dataset CIFAR100 --network_structure resnet152 \
	--learning_rate 0.1 --epochs 200 \
	--regrow_method CH3_L3 --init_mode swi \
	--dim 2 --bias --linearlr --end_factor 0.001 --early_stop --update_interval 1 \
```



SET on ResNet152-CIFAR100 task

```
python run.py \
	--dataset CIFAR100 --network_structure resnet152 \
	--learning_rate 0.1 --epochs 200 \
	--regrow_method random --init_mode kaiming \
	--dim 2 --bias --linearlr --end_factor 0.001 --update_interval 1 \
```



#### Citation

----

If you use our code, please consider to cite:

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

