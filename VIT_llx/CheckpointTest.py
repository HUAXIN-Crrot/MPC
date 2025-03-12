from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageClassification

# 加载图片
img = Image.open('image.png')

# 使用 transformers 加载模型
model_name = "./weights/timm/vit_pwee_patch16_reg1_gap_256.sbb_in1k"  # ViT 模型的名称，替换为你需要的模型名称
processor = AutoProcessor.from_pretrained(model_name)  # 加载预处理器
model = AutoModelForImageClassification.from_pretrained(model_name)  # 加载预训练模型

# 将图像输入到模型的预处理器中，进行适当的调整、归一化等
inputs = processor(images=img, return_tensors="pt")

# 推理
with torch.no_grad():
    outputs = model(**inputs)

# 获取 logits 并计算预测的类别
logits = outputs.logits  # 这是模型的输出 logits
probabilities = torch.nn.functional.softmax(logits, dim=-1)  # 计算 softmax 得到概率分布

# 获取 top 5 类别的概率和索引
top5_probabilities, top5_class_indices = torch.topk(probabilities, k=5)

# 打印 top 5 类别的概率和类别索引
print("Top 5 probabilities: ", top5_probabilities)
print("Top 5 class indices: ", top5_class_indices)

# 获取类别标签（通过 model.config.id2label 映射类别索引到标签名称）
id2label = model.config.id2label  # id2label 是一个字典，映射类别索引到标签
top5_class_labels = [id2label[idx.item()] for idx in top5_class_indices[0]]  # 获取 top5 的标签

# 打印 top 5 类别名称
print("Top 5 class labels: ", top5_class_labels)
