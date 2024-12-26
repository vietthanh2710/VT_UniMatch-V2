import numpy as np
import logging
import os
import torch

class SegmentationMetrics(object):
    r"""Calculate common metrics in semantic segmentation to evalueate model preformance.

    Supported metrics: Pixel accuracy, Dice Coeff, precision score and recall score.
    
    Pixel accuracy measures how many pixels in a image are predicted correctly.

    Dice Coeff is a measure function to measure similarity over 2 sets, which is usually used to
    calculate the similarity of two samples. Dice equals to f1 score in semantic segmentation tasks.
    
    It should be noted that Dice Coeff and Intersection over Union are highly related, so you need 
    NOT calculate these metrics both, the other can be calcultaed directly when knowing one of them.

    Precision describes the purity of our positive detections relative to the ground truth. Of all
    the objects that we predicted in a given image, precision score describes how many of those objects
    actually had a matching ground truth annotation.

    Recall describes the completeness of our positive predictions relative to the ground truth. Of
    all the objected annotated in our ground truth, recall score describes how many true positive instances
    we have captured in semantic segmentation.

    Args:
        eps: float, a value added to the denominator for numerical stability.
            Default: 1e-5

        average: bool. Default: ``True``
            When set to ``True``, average Dice Coeff, precision and recall are
            returned. Otherwise Dice Coeff, precision and recall of each class
            will be returned as a numpy array.

        ignore_background: bool. Default: ``True``
            When set to ``True``, the class will not calculate related metrics on
            background pixels. When the segmentation of background pixels is not
            important, set this value to ``True``.

        activation: [None, 'none', 'softmax' (default), 'sigmoid', '0-1']
            This parameter determines what kind of activation function that will be
            applied on model output.

    Input:
        y_true: :math:`(N, H, W)`, torch tensor, where we use int value between (0, num_class - 1)
        to denote every class, where ``0`` denotes background class.
        y_pred: :math:`(N, C, H, W)`, torch tensor.

    Examples::
        >>> metric_calculator = SegmentationMetrics(average=True, ignore_background=True)
        >>> pixel_accuracy, dice, precision, recall = metric_calculator(y_true, y_pred)
    """
    def __init__(self, eps=1e-5, average=True, ignore_background=True, activation='0-1',ignore_index = 255):
        self.eps = eps
        self.average = average
        self.ignore = ignore_background
        self.activation = activation
        self.ignore_index = ignore_index

    @staticmethod
    def _one_hot(gt, pred, class_num, ignore_index):
        # transform sparse mask into one-hot mask
        # shape: (B, H, W) -> (B, C, H, W)
        pred[np.where(gt == ignore_index)[0]] = ignore_index
        input_shape = tuple(gt.shape)  # (N, H, W, ...)
        new_shape = (input_shape[0], class_num) + input_shape[1:]
        one_hot = torch.zeros(new_shape).to(pred.device, dtype=torch.float)
        target = one_hot.scatter_(1, gt.unsqueeze(1).long().data, 1.0)
        return target

    @staticmethod
    def _get_class_data(gt_onehot, pred, class_num):
        # perform calculation on a batch
        # for precise result in a single image, plz set batch size to 1
        matrix = np.zeros((3, class_num))

        # calculate tp, fp, fn per class
        for i in range(class_num):
            # pred shape: (N, H, W)
            class_pred = pred[:, i, :, :]
            # gt shape: (N, H, W), binary array where 0 denotes negative and 1 denotes positive
            class_gt = gt_onehot[:, i, :, :]

            pred_flat = class_pred.contiguous().view(-1, )  # shape: (N * H * W, )
            gt_flat = class_gt.contiguous().view(-1, )  # shape: (N * H * W, )

            tp = torch.sum(gt_flat * pred_flat)
            fp = torch.sum(pred_flat) - tp
            fn = torch.sum(gt_flat) - tp

            matrix[:, i] = tp.item(), fp.item(), fn.item()

        return matrix

    def _calculate_multi_metrics(self, gt, pred, class_num):
        # calculate metrics in multi-class segmentation
        matrix = self._get_class_data(gt, pred, class_num)
        if self.ignore:
            matrix = matrix[:, 1:]
    
        # tp = np.sum(matrix[0, :])
        # fp = np.sum(matrix[1, :])
        # fn = np.sum(matrix[2, :])

        pixel_acc = (np.sum(matrix[0, :]) + self.eps) / (np.sum(matrix[0, :]) + np.sum(matrix[1, :]))
        dice = (2 * matrix[0] + self.eps) / (2 * matrix[0] + matrix[1] + matrix[2] + self.eps)
        precision = (matrix[0] + self.eps) / (matrix[0] + matrix[1] + self.eps)
        recall = (matrix[0] + self.eps) / (matrix[0] + matrix[2] + self.eps)

        if self.average:
            dice = np.average(dice)
            precision = np.average(precision)
            recall = np.average(recall)

        return pixel_acc, dice, precision, recall

    def __call__(self, y_true, y_pred):
        class_num = y_pred.size(1)

        if self.activation in [None, 'none']:
            activation_fn = lambda x: x
            activated_pred = activation_fn(y_pred)
        elif self.activation == "sigmoid":
            activation_fn = nn.Sigmoid()
            activated_pred = activation_fn(y_pred)
        elif self.activation == "softmax":
            activation_fn = nn.Softmax(dim=1)
            activated_pred = activation_fn(y_pred)
        elif self.activation == "0-1":
            pred_argmax = torch.argmax(y_pred, dim=1)
            activated_pred = self._one_hot(pred_argmax, y_pred, class_num, ignore_index = self.ignore_index)
        else:
            raise NotImplementedError("Not a supported activation!")

        gt_onehot = self._one_hot(y_true, y_pred, class_num,ignore_index=self.ignore_index)
        pixel_acc, dice, precision, recall = self._calculate_multi_metrics(gt_onehot, activated_pred, class_num)
        return pixel_acc, dice, precision, recall


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255, 0, 0])
        cmap[13] = np.array([0, 0, 142])
        cmap[14] = np.array([0, 0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0, 0, 230])
        cmap[18] = np.array([119, 11, 32])

    elif dataset == 'ade20k':
        cmap[0] = np.array([120, 120, 120])
        cmap[1] = np.array([180, 120, 120])
        cmap[2] = np.array([6, 230, 230])
        cmap[3] = np.array([80, 50, 50])
        cmap[4] = np.array([4, 200, 3])
        cmap[5] = np.array([120, 120, 80])
        cmap[6] = np.array([140, 140, 140])
        cmap[7] = np.array([204, 5, 255])
        cmap[8] = np.array([230, 230, 230])
        cmap[9] = np.array([4, 250, 7])
        cmap[10] = np.array([224, 5, 255])
        cmap[11] = np.array([235, 255, 7])
        cmap[12] = np.array([150, 5, 61])
        cmap[13] = np.array([120, 120, 70])
        cmap[14] = np.array([8, 255, 51])
        cmap[15] = np.array([255, 6, 82])
        cmap[16] = np.array([143, 255, 140])
        cmap[17] = np.array([204, 255, 4])
        cmap[18] = np.array([255, 51, 7])
        cmap[19] = np.array([204, 70, 3])
        cmap[20] = np.array([0, 102, 200])
        cmap[21] = np.array([61, 230, 250])
        cmap[22] = np.array([255, 6, 51])
        cmap[23] = np.array([11, 102, 255])
        cmap[24] = np.array([255, 7, 71])
        cmap[25] = np.array([255, 9, 224])
        cmap[26] = np.array([9, 7, 230])
        cmap[27] = np.array([220, 220, 220])
        cmap[28] = np.array([255, 9, 92])
        cmap[29] = np.array([112, 9, 255])
        cmap[30] = np.array([8, 255, 214])
        cmap[31] = np.array([7, 255, 224])
        cmap[32] = np.array([255, 184, 6])
        cmap[33] = np.array([10, 255, 71])
        cmap[34] = np.array([255, 41, 10])
        cmap[35] = np.array([7, 255, 255])
        cmap[36] = np.array([224, 255, 8])
        cmap[37] = np.array([102, 8, 255])
        cmap[38] = np.array([255, 61, 6])
        cmap[39] = np.array([255, 194, 7])
        cmap[40] = np.array([255, 122, 8])
        cmap[41] = np.array([0, 255, 20])
        cmap[42] = np.array([255, 8, 41])
        cmap[43] = np.array([255, 5, 153])
        cmap[44] = np.array([6, 51, 255])
        cmap[45] = np.array([235, 12, 255])
        cmap[46] = np.array([160, 150, 20])
        cmap[47] = np.array([0, 163, 255])
        cmap[48] = np.array([140, 140, 140])
        cmap[49] = np.array([250, 10, 15])
        cmap[50] = np.array([20, 255, 0])
        cmap[51] = np.array([31, 255, 0])
        cmap[52] = np.array([255, 31, 0])
        cmap[53] = np.array([255, 224, 0])
        cmap[54] = np.array([153, 255, 0])
        cmap[55] = np.array([0, 0, 255])
        cmap[56] = np.array([255, 71, 0])
        cmap[57] = np.array([0, 235, 255])
        cmap[58] = np.array([0, 173, 255])
        cmap[59] = np.array([31, 0, 255])
        cmap[60] = np.array([11, 200, 200])
        cmap[61] = np.array([255, 82, 0])
        cmap[62] = np.array([0, 255, 245])
        cmap[63] = np.array([0, 61, 255])
        cmap[64] = np.array([0, 255, 112])
        cmap[65] = np.array([0, 255, 133])
        cmap[66] = np.array([255, 0, 0])
        cmap[67] = np.array([255, 163, 0])
        cmap[68] = np.array([255, 102, 0])
        cmap[69] = np.array([194, 255, 0])
        cmap[70] = np.array([0, 143, 255])
        cmap[71] = np.array([51, 255, 0])
        cmap[72] = np.array([0, 82, 255])
        cmap[73] = np.array([0, 255, 41])
        cmap[74] = np.array([0, 255, 173])
        cmap[75] = np.array([10, 0, 255])
        cmap[76] = np.array([173, 255, 0])
        cmap[77] = np.array([0, 255, 153])
        cmap[78] = np.array([255, 92, 0])
        cmap[79] = np.array([255, 0, 255])
        cmap[80] = np.array([255, 0, 245])
        cmap[81] = np.array([255, 0, 102])
        cmap[82] = np.array([255, 173, 0])
        cmap[83] = np.array([255, 0, 20])
        cmap[84] = np.array([255, 184, 184])
        cmap[85] = np.array([0, 31, 255])
        cmap[86] = np.array([0, 255, 61])
        cmap[87] = np.array([0, 71, 255])
        cmap[88] = np.array([255, 0, 204])
        cmap[89] = np.array([0, 255, 194])
        cmap[90] = np.array([0, 255, 82])
        cmap[91] = np.array([0, 10, 255])
        cmap[92] = np.array([0, 112, 255])
        cmap[93] = np.array([51, 0, 255])
        cmap[94] = np.array([0, 194, 255])
        cmap[95] = np.array([0, 122, 255])
        cmap[96] = np.array([0, 255, 163])
        cmap[97] = np.array([255, 153, 0])
        cmap[98] = np.array([0, 255, 10])
        cmap[99] = np.array([255, 112, 0])
        cmap[100] = np.array([143, 255, 0])
        cmap[101] = np.array([82, 0, 255])
        cmap[102] = np.array([163, 255, 0])
        cmap[103] = np.array([255, 235, 0])
        cmap[104] = np.array([8, 184, 170])
        cmap[105] = np.array([133, 0, 255])
        cmap[106] = np.array([0, 255, 92])
        cmap[107] = np.array([184, 0, 255])
        cmap[108] = np.array([255, 0, 31])
        cmap[109] = np.array([0, 184, 255])
        cmap[110] = np.array([0, 214, 255])
        cmap[111] = np.array([255, 0, 112])
        cmap[112] = np.array([92, 255, 0])
        cmap[113] = np.array([0, 224, 255])
        cmap[114] = np.array([112, 224, 255])
        cmap[115] = np.array([70, 184, 160])
        cmap[116] = np.array([163, 0, 255])
        cmap[117] = np.array([153, 0, 255])
        cmap[118] = np.array([71, 255, 0])
        cmap[119] = np.array([255, 0, 163])
        cmap[120] = np.array([255, 204, 0])
        cmap[121] = np.array([255, 0, 143])
        cmap[122] = np.array([0, 255, 235])
        cmap[123] = np.array([133, 255, 0])
        cmap[124] = np.array([255, 0, 235])
        cmap[125] = np.array([245, 0, 255])
        cmap[126] = np.array([255, 0, 122])
        cmap[127] = np.array([255, 245, 0])
        cmap[128] = np.array([10, 190, 212])
        cmap[129] = np.array([214, 255, 0])
        cmap[130] = np.array([0, 204, 255])
        cmap[131] = np.array([20, 0, 255])
        cmap[132] = np.array([255, 255, 0])
        cmap[133] = np.array([0, 153, 255])
        cmap[134] = np.array([0, 41, 255])
        cmap[135] = np.array([0, 255, 204])
        cmap[136] = np.array([41, 0, 255])
        cmap[137] = np.array([41, 255, 0])
        cmap[138] = np.array([173, 0, 255])
        cmap[139] = np.array([0, 245, 255])
        cmap[140] = np.array([71, 0, 255])
        cmap[141] = np.array([122, 0, 255])
        cmap[142] = np.array([0, 255, 184])
        cmap[143] = np.array([0, 92, 255])
        cmap[144] = np.array([184, 255, 0])
        cmap[145] = np.array([0, 133, 255])
        cmap[146] = np.array([255, 214, 0])
        cmap[147] = np.array([25, 194, 194])
        cmap[148] = np.array([102, 255, 0])
        cmap[149] = np.array([92, 0, 255])

    return cmap
