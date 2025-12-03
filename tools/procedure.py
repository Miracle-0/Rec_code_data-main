"""
Design training and test process
"""
import time
from tools import world, utils
import numpy as np
import torch
import multiprocessing
from dataset import dataloader
from tools.world import cprint

CORES = multiprocessing.cpu_count() // 2  # 可根据机器调整

def Train(dataset: dataloader.Loader, recommend_model, loss_class, epoch, config, w=None, sampling_strategy="standard"):
    """
    Train function with support for standard and noise-aware negative sampling.

    Args:
        dataset: 数据加载器
        recommend_model: 推荐模型
        loss_class: 损失函数
        epoch: 当前训练轮次
        config: 配置字典
        w: TensorBoard writer
        sampling_strategy: 负采样策略 ("standard" 或 "noise")
    """
    Recmodel = recommend_model
    Recmodel.train()
    loss = loss_class

    start = time.time()

    users, posItems = dataset.trainUser_tensor, dataset.trainItem_tensor
    users, posItems = utils.shuffle(users, posItems)

    batch_size = config["train_batch"]
    total_batch = len(users) // batch_size + 1
    aver_loss = 0.0

    iter_num = epoch * total_batch
    for batch_id, (batch_users, batch_pos) in enumerate(utils.minibatch(users, posItems, batch_size=batch_size)):
        batch_users = batch_users.cuda(non_blocking=True)
        batch_pos = batch_pos.cuda(non_blocking=True)

        if sampling_strategy == "standard":
            # 标准负采样：严格从非交互物品中采样 (~interaction)
            batch_not_interaction_tensor = (~dataset.interaction_tensor[batch_users]).float()
            batch_neg = torch.multinomial(batch_not_interaction_tensor, config["num_negative_items"], replacement=True)
        elif sampling_strategy == "noise":
            # 噪声负采样：人为引入噪声以测试鲁棒性
            # 策略：混合策略。大部分样本来自“未交互项”（标准负样本），
            # 少部分样本来自“全集随机项”（可能包含正样本，即引入了 False Negative 噪声）
            
            num_neg = config["num_negative_items"]
            noise_ratio = config.get("noise_ratio", 0.2) # 默认引入 20% 的噪声，可从 config 读取
            
            num_noise = int(num_neg * noise_ratio)
            num_clean = num_neg - num_noise

            # 1. 清洁部分 (Standard Negative Sampling)
            batch_not_interaction_tensor = (~dataset.interaction_tensor[batch_users]).float()
            batch_neg_clean = torch.multinomial(batch_not_interaction_tensor, num_clean, replacement=True)

            # 2. 噪声部分 (Random Sampling - 模拟噪声干扰)
            # 直接在所有物品 ID 范围内随机采样，不避开用户喜欢的物品
            m_items = dataset.interaction_tensor.size(1)
            batch_neg_noise = torch.randint(0, m_items, (len(batch_users), num_noise)).cuda()

            # 3. 拼接
            batch_neg = torch.cat([batch_neg_clean, batch_neg_noise], dim=1)
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

        # ================= 验证代码开始 (调试完成后可删除) =================
        # if batch_id == 0: # 只在每个 epoch 的第一个 batch 检查，避免刷屏
            # 获取当前 batch 用户的真实交互历史 (1 表示交互过，0 表示未交互)
            # 注意：dataset.interaction_tensor 通常在 CPU 上，需要转到 GPU 才能和 batch_neg (GPU) 运算
            current_user_history = dataset.interaction_tensor[batch_users].cuda()
            
            # 使用 gather 函数，检查 batch_neg 对应的索引在历史记录中是否为 1
            # batch_neg 形状: [batch_size, num_neg]
            # hits 形状: [batch_size, num_neg]，如果某位置是 1，说明该负样本其实是正样本（噪声）
            hits = torch.gather(current_user_history, 1, batch_neg)
            
            noise_count = hits.sum().item()
            total_samples = hits.numel()
            noise_rate = noise_count / total_samples
            
            print(f"\n[验证采样策略: {sampling_strategy}]")
            print(f"  - 负样本总数: {total_samples}")
            print(f"  - 命中正样本数 (False Negatives): {int(noise_count)}")
            print(f"  - 实际噪声率: {noise_rate:.4%}")
            
            if sampling_strategy == "standard":
                if noise_count == 0:
                    print("  -> 验证通过：标准采样未引入噪声。")
                else:
                    print("  -> 警告：标准采样中发现了噪声！代码可能存在 Bug。")
            elif sampling_strategy == "noise":
                if noise_count > 0:
                    print("  -> 验证通过：成功引入了噪声。")
                else:
                    print("  -> 提示：当前 Batch 未命中噪声（如果噪声比例很低，这可能是正常的，多跑几轮看看）。")
            print("="*60 + "\n")
        # ================= 验证代码结束 =================

        cri = loss.step(batch_users, batch_pos, batch_neg)
        if w is not None:
            w.add_scalar("Train/Loss_iter", cri, iter_num + batch_id)
        aver_loss += cri

    aver_loss = aver_loss / total_batch
    if w is not None:
        w.add_scalar("Train/Loss_epoch", aver_loss, epoch)
    time_one_epoch = int(time.time() - start)
    return f"Loss{aver_loss:.3f}-Time{time_one_epoch}"

def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg, hitratio = [], [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret["precision"])
        recall.append(ret["recall"])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
        hitratio.append(utils.HitRatio(r))
    return {
        "recall": np.array(recall),
        "precision": np.array(pre),
        "ndcg": np.array(ndcg),
        "hitratio": np.array(hitratio),
    }


def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config["test_u_batch_size"]
    testDict: dict = dataset.testDict

    Recmodel = Recmodel.eval()
    max_K = max(world.topks)

    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {
        "precision": np.zeros(len(world.topks)),
        "recall": np.zeros(len(world.topks)),
        "ndcg": np.zeros(len(world.topks)),
        "hitratio": np.zeros(len(world.topks)),
    }

    with torch.no_grad():
        users = list(testDict.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1

        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long().cuda()

            rating = Recmodel.getUsersRating(batch_users_gpu)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)

            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)

        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = [test_one_batch(x) for x in X]

        for result in pre_results:
            results["recall"] += result["recall"]
            results["precision"] += result["precision"]
            results["ndcg"] += result["ndcg"]
            results["hitratio"] += result["hitratio"]

        results["recall"] /= float(len(users))
        results["precision"] /= float(len(users))
        results["ndcg"] /= float(len(users))
        results["hitratio"] /= float(dataset.testDataSize)

        # 写入 TensorBoard
        if w is not None:
            for i, k in enumerate(world.topks):
                w.add_scalar(f"Test/Recall@{k}", results["recall"][i], epoch)
                w.add_scalar(f"Test/Precision@{k}", results["precision"][i], epoch)
                w.add_scalar(f"Test/NDCG@{k}", results["ndcg"][i], epoch)
                w.add_scalar(f"Test/HitRatio@{k}", results["hitratio"][i], epoch)

        if multicore == 1:
            pool.close()

        return results