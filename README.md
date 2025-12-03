# Rec_code_data

## Code
The code implement the models like LightGCN, MF, LightGCL, XSimGCL, SimGCL, and implement various loss function like BPR Loss, BCE Loss, Softmax Loss. Pay attention to how to implement in an efficient way, such as negative sampling.

## Data
Each dataset contains two txt files, representing the test set and the training set respectively. The number 'k' after '_' in filename represents k-core processed. In txt file, the first column represents the user ID, and the second column represents the item ID. All IDs are serialized.

待办:
tools/world.py：添加新参数 noise_ratio / cap_eta / eval_interval
tools/procedure.py：实现噪声注入逻辑 + 写指标到 TensorBoard
optimizer/optim_CappedSoftmax.py：实现 Capped Softmax Loss
可选：utils.py 添加新指标（MRR@K）
写一个 run script 批量跑不同 η 和 noise_ratio（后续）

如何运行代码：
1. 激活现有的conda环境：conda activate recenv
2. 跑测试代码
python main.py --model=mf --dataset=gowalla_10 --loss=softmax --trainbatch 1024 --testbatch 1024 --epochs 100 --cuda 0 --recdim 64 --comment="mf_eval" --experiment="noise_standard" --noise_level=0.2

python main.py --model=mf --dataset=gowalla_10 --loss=softmax --trainbatch 1024 --testbatch 1024 --epochs 100 --cuda 0 --recdim 64 --comment="mf_eval"  

3. 查看结果：tensorboard --logdir .\log\gowalla_10 

dataset: 里面有各种数据集
model：里面是各种推荐模型算法