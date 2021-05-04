CUDA_VISIBLE_DEVICES=1 python mainTTT.py \
--tag 'txt_bertopt' --model_type 'nothing' \
--version 'v1.0-trainval' --data_type 'real' \
--batch_size 8 --num_epochs 4000 --agent_embed_dim 128 \
--ploss_type 'map' --path_ploss_type 'mseloss' --num_candidates 10 \
--beta 0.1 --learning_rate 0.001 --num_workers 20  --path_weight 0.5 --load_dir '/home/user/datasets/cmu_dataset'
