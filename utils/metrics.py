# Model validation metrics
"""
该文件通过获得到的预测结果与ground truth表现计算指标P、R、F1-score、AP、不同阈值下的mAP等。
同时，该文件将上述指标进行了可视化，绘制了混淆矩阵以及P-R曲线
"""
from pathlib import Path #调用路径操作模块

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import general#从当前文件所处的相对路径调用general.py

#fitness函数 通过指标加权的形式返回适应度
def fitness(x):
    # 以矩阵的加权组合作为模型的适应度
    w = [0.0, 0.0, 0.1, 0.9]  # 每个变量对应的权重 [P, R, mAP@0.5, mAP@0.5:0.95]
    # (torch.tensor).sum(1) 每一行求和tensor为二维时返回一个以每一行求和为结果的行向量
    return (x[:, :4] * w).sum(1)

#ap_per_class 函数计算每一个类的AP指标
def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """ 计算平均精度（AP），并绘制P-R曲线
        源代码来源: https://github.com/rafaelpadilla/Object-Detection-Metrics.
        # Arguments（变量）
            tp:  True positives (nparray, nx1 or nx10).   真阳
            conf:  Objectness value from 0-1 (nparray).   目标的置信度取值0-1
            pred_cls:  Predicted object classes (nparray).预测目标类别
            target_cls:  True object classes (nparray).   真实目标类别
            plot:  Plot precision-recall curve at mAP@0.5 是否绘制P-R曲线 在mAP@0.5的情况下
            save_dir:  P-R曲线图的保存路径
        # Returns（返回）
            像faster-rcnn那种方式计算AP （这里涉及计算AP的两种不同方式 建议查询）
            The average precision as computed in py-faster-rcnn.
        """
    # 将目标进行排序
    # np.argsort(-conf)函数返回一个索引数组 其中每一个数按照conf中元素从大到小 置为 0,1...n
    i = np.argsort(-conf)
    # tp conf pred_cls 三个矩阵均按照置信度从大到小进行排列
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # 找到各个独立的类别
    # np.unique()会返回输入array中出现至少一次的变量 这里返回所有独立的类别
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # 创建P-R曲线 并 计算每一个类别的AP
    #px, py = np.linspace(0, 1, 1000), []  # for plotting
    px, py = np.linspace(0, 10, 1000), []  # for plotting
    # 初始化 对每一个类别在每一个IOU阈值下面 计算P R AP参数
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):# ci为类别对应索引 c为具体的类别
        # i为一个包含True/False 的列表 代表 pred_cls array 各元素是否与 类别c 相同
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # ground truth中 类别c 的个数 all_results
        n_p = i.sum()  # 预测类别中为 类别c 的个数

        if n_p == 0 or n_l == 0:#如果没有预测到 或者 ground truth没有标注 则略过类别c
            continue
        else:

            """ 
                        计算 FP（False Positive） 和 TP(Ture Positive)
                        tp[i] 会根据i中对应位置是否为False来决定是否删除这一位的内容，如下所示：
                        a = np.array([0,1,0,1]) i = np.array([True,False,False,True]) b = a[i]
                        则b为：[0 1]
                        而.cumsum(0)函数会 按照对象进行累加操作，如下所示：
                        a = np.array([0,1,0,1]) b = a.cumsum(0)
                        则b为：[0,1,1,2]
                        （FP + TP = all_detections 所以有 fp[i] = 1 - tp[i]）
                        所以fpc为 类别c 按照置信度从大到小排列 截止到每一位的FP数目
                            tpc为 类别c 按照置信度从大到小排列 截止到每一位的TP数目
                        recall 和 precision 均按照元素从小到大排列
                        """
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            # Recall = TP / (TP + FN) = TP / all_results = TP / n_l
            recall = tpc / (n_l + 1e-16) # 加一个1e-16的目的是防止n_l为0 时除不开
            """
                        np.interp() 函数第一个输入值为数值 第二第三个变量为一组x y坐标 返回结果为一个数值
                        这个数值为 找寻该数值左右两边的x值 并将两者对应的y值取平均 如果在左侧或右侧 则取 边界值
                        如果第一个输入为数组 则返回一个数组 其中每一个元素按照上述计算规则产生
                        """
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0) # pr_score 处的y值

            # Precision
            # Precision = TP / TP + FP = TP / all_detections
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)   # pr_score 处的y值

            # 从P-R曲线中计算AP
            for j in range(tp.shape[1]): #这里对每一个IOU阈值 下的参数进行计算
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j]) #取每一个阈值计算AP
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # mAP@0.5处的P

    #计算F1分数 P和R的调和平均值
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')

#compute_ap 通过输入P和R的值来计算AP
def compute_ap(recall, precision):#计算AP
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # 在开头和末尾添加保护值 防止全零的情况出现
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))
    """
        此处需要关注precision列表输入时元素为从小到大排列（由上一个函数）
        np.filp()函数会把一维数组每个元素的顺序进行翻转 第一个翻转成为最后一个
        np.maximum.accumulate() 函数会返回输入
        mpre = np.flip(np.maximum.accumulate(np.flip(recall)))
        Q?：此处mpre返回的是是否由输入数组中最大的元素组成的数组如
        recall = np.array([0.1,0.2,0.2,0.3,0.4])
        final_1 = np.flip(np.maximum.accumulate(np.flip(recall)))
        final_2 = np.flip(np.maximum.accumulate(recall))
        final_1：[0.4 0.4 0.4 0.4 0.4]
        final_2：[0.4 0.3 0.2 0.2 0.1]
        """
    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':#计算 AP 的方法为间断性的
        # x 为0-1 101个点组成的等差数列数组 为间断点
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        # np.trapz(list,list) 计算两个list对应点与点之间四边形的面积 以定积分形式估算AP
        # 按照P-R曲线的定义 R近似为递增数组 P为近似递减数组 如上中final_2结果
        ap = np.trapz(np.interp(x, mrec, mpre), x)   # 前一个数组为纵坐标 第二个为横坐标
    else:  # 'continuous' #采用连续的方法计算AP
        """
                通过错位的方式 判断哪个点发生了改变并通过！=判断 返回一个布尔数组 
                再通过np.where()函数找出 mrec中对应发生的改变点 i为一个数组 每一个
                元素代表当前位置到下一个位置发生改变
                """
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

#ConfusionMatrix 类为求解混淆矩阵并进行绘图
class ConfusionMatrix:# nc为训练的类别 conf为置信度 iou_thres 为IOU loss的阈值
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        返回 各个box之间的交并比(iou)
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        每一个box的集合都被期望使用(x1,y1,x2,y2)的形式 这两个点为box的对角顶点
        Arguments: detections 和 labels的数据结构
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
            无返回 更新混淆矩阵
        """
        # detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        detections = detections[detections[:, 4] > self.conf]# 返回检测大于阈值的框
        # gt_classes (Array[M, 1]), ground_truth class
        gt_classes = labels[:, 0].int()# 返回ground truth的类别
        detection_classes = detections[:, 5].int() # 返回检测到的类别
        # iou计算	box1 (Array[N, 4]), x1, y1, x2, y2
        #           box2 (Array[M, 4]), x1, y1, x2, y2
        # iou (Tensor[N, M]) NxM矩阵包含了 box1中每一个框和box2中每一个框的iou值
        # 非常重要！ iou中坐标 (n1,m1) 代表 第n1个ground truth 框 和 第m1个 预测框的
        iou = general.box_iou(labels[:, 1:], detections[:, :4])#调用general中计算iou的方式计算iou
        # x为一个含有两个tensor的tuple表示iou中大于阈值的值的坐标，第一个tensor为第几行，第二个为第几列
        x = torch.where(iou > self.iou_thres)#找到iou中大于阈值的那部分并提取
        if x[0].shape[0]:# 当大于阈值的坐标不止一个的时候
            """
                        torch.cat(inputs,dimension=0) 为在指定的维度对 张量inputs进行堆叠 
                        二维情况下 0代表按照行 1代表按照列 0时会增加行 1时会增加列
                        torch.stack(x,1) 当x为二维张量的时候 本质上是对x做转置操作
                        .cpu()是将变量转移到cpu上进行运算.numpy()是转换为numpy数组
                        matches (Array[N, 3]), row,col,iou_value ！！！
                                row为大于阈值的iou张量中点的横坐标 col为纵坐标 iou_value为对应的iou值
                        """
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))# 这里返回一个0行3列全0的二维数组 ？因为没有一个例子满足这个要求

        n = matches.shape[0] > 0#这里n为 True 或 False 用于判断是否存在满足阈值要求的对象是否至少有一个
        """
                a.transpose()是numpy中轮换维度索引的方法 对二维数组表示为转置
                此处matches (Array[N, 3]), row,col,iou_value
                物理意义：在大于阈值的前提下，N*M种label与预测框的组合可能下，每一种预测框与所有label框iou值最大的那个
                m0，m1  (Array[1, N])
                m0代表 满足上述条件的第i个label框   （也即类别）
                m1代表 满足上述条件的第j个predict框 （也即类别）
                """
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):#解析ground truth 中的类别
            j = m0 == i
            if n and sum(j) == 1:#检测到的目标至少有1个 且 groundtruth对应只有一个
                self.matrix[detection_classes[m1[j]], gc] += 1  # TP 判断正确的数目加1
            else:
                self.matrix[self.nc, gc] += 1  # 背景 FP（false positive） 个数加1 背景被误认为目标

        if n:# 当目标不止一个时
            for i, dc in enumerate(detection_classes):# i为索引 dc为每一个目标检测到的类别
                if not any(m1 == i):# 检测到目标 但是目标与groundtruth的iou小于之前要求的阈值则
                    self.matrix[dc, self.nc] += 1  # 背景 FN 个数加1 （目标被检测成了背景）

    def matrix(self): #返回matrix变量 该matrix为混淆矩阵
        return self.matrix

    def plot(self, save_dir='', names=()):
        try:
            import seaborn as sn#seaborn 为易于可视化的一个模块

            array = self.matrix / (self.matrix.sum(0).reshape(1, self.nc + 1) + 1E-6)  # 矩阵归一化为0-1
            array[array < 0.005] = np.nan  # 小于0.005的值被认为NaN

            fig = plt.figure(figsize=(12, 9), tight_layout=True) #初始化画布
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)   # 设置标签的尺寸
            labels = (0 < len(names) < 99) and len(names) == self.nc  # 用于绘制过程中判断是否应用names
            # 绘制热力图 即混淆矩阵可视化
            sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names + ['background FP'] if labels else "auto",
                       yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            # 下三行代码为设置figure的横坐标 纵坐标及保存该图片
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        except Exception as e:
            pass

    def print(self):# 打印出每一个元素对应的数据
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    #fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 3), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
