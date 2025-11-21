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

def Train(dataset: dataloader.Loader, recommend_model, loss_class, epoch, config, w=None):
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

        batch_not_interaction_tensor = (~dataset.interaction_tensor[batch_users]).float()
        batch_neg = torch.multinomial(batch_not_interaction_tensor, config["num_negative_items"], replacement=True)
        cri = loss.step(batch_users, batch_pos, batch_neg)
        if w is not None:
            w.add_scalar("Train/Loss_iter", cri, iter_num + batch_id)
        aver_loss += cri

    aver_loss = aver_loss / total_batch
    if w is not None:
        w.add_scalar("Train/Loss_epoch", aver_loss, epoch)
    time_one_epoch = int(time.time() - start)
    return f"Loss{aver_loss:.3f}-Time{time_one_epoch}"

# def test_one_batch(X):
#     sorted_items = X[0].numpy()
#     groundTrue = X[1]
#     r = utils.getLabel(groundTrue, sorted_items)
#     pre, recall, ndcg, hitratio = [], [], [], []
#     for k in world.topks:
#         ret = utils.RecallPrecision_ATk(groundTrue, r, k)
#         pre.append(ret["precision"])
#         recall.append(ret["recall"])
#         ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
#         hitratio.append(utils.HitRatio(r))
#     return {
#         "recall": np.array(recall),
#         "precision": np.array(pre),
#         "ndcg": np.array(ndcg),
#         "hitratio": np.array(hitratio),
#     }

    # 原始负采样代码片段（简化示例）
    batch_not_interaction_tensor = (~dataset.interaction_tensor[batch_users]).float()
    batch_neg = torch.multinomial(
        batch_not_interaction_tensor, 
        config["num_negative_items"], 
        replacement=True
    )  # shape: [batch_size, num_negative_items]

    # 新增：带噪声负采样
    if config.get("noise_ratio", 0) > 0:
        noise_ratio = config["noise_ratio"]
        # 对每个用户注入已知正样本
        noisy_batch_neg = batch_neg.clone()
        for local_idx, u in enumerate(batch_users):
            pos_items = dataset.getUserPosItems([u])[0]  # list of this user's train positives
            if len(pos_items) == 0:
                continue
            num_to_inject = int(len(noisy_batch_neg[local_idx]) * noise_ratio)
            num_to_inject = max(1, num_to_inject) if noise_ratio > 0 else 0
            if num_to_inject > 0:
                # 选取若干正样本
                inject_pos = np.random.choice(pos_items, size=min(num_to_inject, len(pos_items)), replace=False)
                inject_pos = torch.tensor(inject_pos, device=noisy_batch_neg.device)
                # 随机替换掉 neg 的前 num_to_inject 个位置 (也可以随机选位置)
                noisy_batch_neg[local_idx, :inject_pos.shape[0]] = inject_pos
        batch_neg = noisy_batch_neg

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