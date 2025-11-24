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
            # 标准负采样
            batch_not_interaction_tensor = (~dataset.interaction_tensor[batch_users]).float()
            batch_neg = torch.multinomial(batch_not_interaction_tensor, config["num_negative_items"], replacement=True)
        elif sampling_strategy == "noise":
            # 噪声负采样
            batch_not_interaction_tensor = (~dataset.interaction_tensor[batch_users]).float()
            noise = torch.rand_like(batch_not_interaction_tensor) * config.get("noise_level", 0.1)
            noisy_tensor = batch_not_interaction_tensor + noise
            batch_neg = torch.multinomial(noisy_tensor, config["num_negative_items"], replacement=True)
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

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