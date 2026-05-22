# 钢材表面缺陷检测项目 
 
## 项目概述 
本项目基于YOLOv8实现了6种钢材表面缺陷的自动检测。 
 
## 缺陷类型 
1. crazing（裂纹） 
2. inclusion（夹杂物） 
3. patches（斑点） 
4. pitted_surface（麻点） 
5. rolled-in_scale（氧化皮） 
6. scratches（划痕） 
 
## 检测精度 
基线模型在NEU-DET数据集上整体mAP@0.5=0.750。 
各类别精度如下： 
- 裂纹（crazing）基线精度mAP@0.5=0.359，经优化后提升至0.525。 
- 夹杂物（inclusion）精度mAP@0.5=0.880。 
- 斑点（patches）精度mAP@0.5=0.906。 
- 麻点（pitted_surface）精度mAP@0.5=0.848。 
- 氧化皮（rolled-in_scale）精度mAP@0.5=0.676。 
- 划痕（scratches）精度mAP@0.5=0.833。 
 
## 优化方法 
针对裂纹类缺陷检测精度低的问题，尝试了高分辨率训练和过采样两种策略。 
最终过采样策略将裂纹精度从0.359提升至0.525。 
