import torch

from mmdet.registry import TASK_UTILS
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import delta2bbox as original_delta2bbox
#可删
def safe_delta2bbox(bboxes,
                    pred_bboxes,
                    means=(0., 0., 0., 0.),
                    stds=(1., 1., 1., 1.),
                    max_shape=None,
                    wh_ratio_clip=16 / 1000,
                    clip_border=True,
                    add_ctr_clamp=False,
                    ctr_clamp=32):
    """增强版的delta2bbox函数，处理尺寸不匹配的情况。

    Args:
        bboxes (torch.Tensor): 基准框，形状为(N, 4)或(B, N, 4)
        pred_bboxes (torch.Tensor): 预测的偏移量，形状为(N, 4)或(B, N, 4)
        means (Sequence[float]): 均值. 默认为(0., 0., 0., 0.)
        stds (Sequence[float]): 标准差. 默认为(1., 1., 1., 1.)
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],可选): 图像的最大尺寸(高度, 宽度).
        wh_ratio_clip (float): 宽高比裁剪的值.
        clip_border (bool, 可选): 是否裁剪超出边界的框. 默认为True
        add_ctr_clamp (bool): 是否对中心点预测进行限制. 默认为False
        ctr_clamp (int): 中心点裁剪的值. 默认为32

    Returns:
        torch.Tensor: 解码后的框
    """
    # 确保输入尺寸一致
    if bboxes.size(0) != pred_bboxes.size(0):
        # 如果bboxes比pred_bboxes多，截取前面部分
        if bboxes.size(0) > pred_bboxes.size(0):
            bboxes = bboxes[:pred_bboxes.size(0)]
        # 如果pred_bboxes比bboxes多，截取前面部分
        else:
            pred_bboxes = pred_bboxes[:bboxes.size(0)]
    
    # 确保两个张量具有相同的设备
    if bboxes.device != pred_bboxes.device:
        pred_bboxes = pred_bboxes.to(bboxes.device)
    
    # 调用原始的delta2bbox函数
    return original_delta2bbox(
        bboxes, pred_bboxes, means, stds, max_shape, 
        wh_ratio_clip, clip_border, add_ctr_clamp, ctr_clamp
    )

@TASK_UTILS.register_module()
class SafeDeltaXYWHBBoxCoder(DeltaXYWHBBoxCoder):
    """安全版本的DeltaXYWHBBoxCoder，使用安全的delta2bbox函数。"""

    def decode(self,
              bboxes,
              pred_bboxes,
              max_shape=None,
              wh_ratio_clip=16 / 1000,
              clip_border=True,
              add_ctr_clamp=False,
              ctr_clamp=32):
        """解码bboxes回归偏移。
        
        使用安全版本的delta2bbox，处理尺寸不匹配问题。
        """
        return safe_delta2bbox(
            bboxes, pred_bboxes, self.means, self.stds, max_shape,
            wh_ratio_clip, clip_border, add_ctr_clamp, ctr_clamp) 