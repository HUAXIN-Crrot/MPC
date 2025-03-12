from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageClassification
from urllib.request import urlopen

# 加载图片
img = Image.open('image.png')

# 加载 processor 和 model
small_model_name = "./weights/timm/vit_pwee_patch16_reg1_gap_256.sbb_in1k"  
large_model_name = "./weights/timm/vit_little_patch16_reg4_gap_256.sbb_in1k"  

# 处理器负责图像预处理
small_processor = AutoProcessor.from_pretrained(small_model_name)
large_processor = AutoProcessor.from_pretrained(large_model_name)

# 加载预训练的 ViT 模型
small_model = AutoModelForImageClassification.from_pretrained(small_model_name)
large_model = AutoModelForImageClassification.from_pretrained(large_model_name)

# 设置模型为评估模式
small_model.eval()
large_model.eval()

# 将图像输入到模型的预处理器中，进行适当的调整、归一化等
small_inputs = small_processor(images=img, return_tensors="pt")
large_inputs = large_processor(images=img, return_tensors="pt")

# 定义协同判断逻辑
def co_predict(small_model, large_model, small_processor, large_processor, img, threshold=0.8):
    # 使用小模型进行预测
    with torch.no_grad():
        small_outputs = small_model(**small_inputs)
        small_probs = torch.nn.functional.softmax(small_outputs.logits, dim=-1)
        
        # 获取最大概率的类别
        small_pred = torch.argmax(small_probs, dim=1).item()
        small_max_prob = small_probs[0, small_pred].item()

        # 如果小模型的概率足够高，直接返回小模型的预测结果
        if small_max_prob > threshold:
            print(f"Small model prediction: {small_pred} with probability {small_max_prob:.4f}")
            return small_pred, small_max_prob
        else:
            print(f"Small model uncertain with probability {small_max_prob:.4f}. Using large model.")
            # 使用大模型进行最终预测
            with torch.no_grad():
                large_outputs = large_model(**large_inputs)
                large_probs = torch.nn.functional.softmax(large_outputs.logits, dim=-1)
                large_pred = torch.argmax(large_probs, dim=1).item()
                large_max_prob = large_probs[0, large_pred].item()
                print(f"Large model prediction: {large_pred} with probability {large_max_prob:.4f}")
                return large_pred, large_max_prob

# 执行协同预测
pred, prob = co_predict(small_model, large_model, small_processor, large_processor, img)

# 打印最终的预测结果
print(f"Final Prediction: {pred} with probability {prob:.4f}")
