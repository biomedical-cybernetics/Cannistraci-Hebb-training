bs=10240
ac=1
dp=0.1
attdp=0.1
gn=0
ws=6000
lr=2
CUDA_VISIBLE_DEVICES=4 python train.py \
		-world_size 1 \
		-gpu_ranks 0 \
	-rnn_size 512 \
		-word_vec_size 512 \
	-transformer_ff 2048 \
		-batch_type tokens \
		-batch_size $bs \
		-accum_count $ac \
		-train_steps 20000 \
		-max_generator_batches 0 \
		-normalization tokens \
		-dropout $dp \
	-attention_dropout $attdp \
		-max_grad_norm $gn \
		-optim adam \
		-encoder_type chttransformer \
		-decoder_type chttransformer \
		-manifold Euclidean \
		-position_encoding \
		-param_init 0 \
		-param_init_glorot \
		-task iwslt \
		-adam_beta2 0.998 \
	-warmup_steps $ws \
		-learning_rate $lr \
	-weight_decay 0 \
		-decay_method inoam \
	--iterative_warmup_steps 20 \
		-label_smoothing 0.1 \
		-data data/iwslt14/iwslt14.tokenized.de-en/processed.noshare \
		-layers 6 \
	-heads 8 \
		-report_every 100 \
		-save_checkpoint_steps 500 \
	-valid_steps 500 \
	-master_port 1345 \
	-keep_checkpoint 10 \
	--use_cht \
	--update_interval 100 \
	--sparsity 0.9 \
	--init_mode swi \
	--regrow_method random \
	--bias --remove_method weight_magnitude
