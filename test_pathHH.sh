CUDA_VISIBLE_DEVICES=2 python main_goalMLP2HH.py \
--model_type 'AttGlobal_Scene_CAM_Goal_NFDecoder' \
--version 'v1.0-trainval' --data_type 'real' \
--ploss_type 'map' \
--beta 0.1 --batch_size 1 --gpu_devices 0 \
--test_times 1 \
--load_dir '/home/user/challenge_6sec' --num_workers 20 --num_candidates 10 \
--test_ckpt '/home/user/jegyeong/NaverLABS/test.pth.tar'  \
--test_dir 'results'