import timm
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer
from transformers import CLIPModel, CLIPConfig
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import logging
import datetime
import torch.nn.functional as F
from timm.models.helpers import load_checkpoint
# from pykan.kan.KAN import KAN # 导入KAN网络
from Sophia.sophia import SophiaG

def read_excel_file_and_encode_labels(file_path, disease_to_label):
    df = pd.read_excel(file_path)
    # 应用映射
    df['encoded_labels'] = df['disease'].map(disease_to_label)
    return df

def find_file_with_extension(image_dir, base_filename, extensions):
    """
    尝试不同的扩展名来找到存在的文件。

    :param image_dir: 存储图像的目录路径
    :param base_filename: 图像文件的基础名称（不含扩展名）
    :param extensions: 尝试的扩展名列表
    :return: 完整的文件路径，如果找不到文件则返回None
    """
    for ext in extensions:
        potential_path = os.path.join(image_dir, f"{base_filename}{ext}")
        if os.path.exists(potential_path):
            return potential_path
    return None

class ImageTextDataset(Dataset):
    def __init__(self, dataframe, image_dir, tokenizer, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row['text']
        label = row['encoded_labels']  # 直接使用预先编码的标签
        image_file = row['file']
        image_path = find_file_with_extension(self.image_dir, image_file, ['.jpg', '.JPG', ''])
        if image_path is None:
            raise FileNotFoundError(f"File {image_file} with extensions .jpg or .JPG not found in {self.image_dir}")
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        encoded_text = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512).input_ids.squeeze(0)

        # 生成负样本
        negative_indices = [i for i in range(len(self.dataframe)) if i != idx]
        negative_samples = np.random.choice(negative_indices, 9, replace=False)
        negative_texts = []
        for neg_idx in negative_samples:
            neg_row = self.dataframe.iloc[neg_idx]
            neg_text = neg_row['text']
            encoded_neg_text = self.tokenizer(neg_text, return_tensors='pt', padding='max_length', truncation=True,
                                              max_length=512).input_ids.squeeze(0)
            negative_texts.append(encoded_neg_text)
        return image, encoded_text, label, negative_texts

class AdaptiveFeatureModule(nn.Module):
    def __init__(self, feature_dim):
        super(AdaptiveFeatureModule, self).__init__()
        self.delta = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, x):
        return x + self.delta

class CrossModalResidualBlock(nn.Module):
    def __init__(self, feature_dim):
        super(CrossModalResidualBlock, self).__init__()
        self.text_to_image = nn.Linear(feature_dim, feature_dim)
        self.image_to_text = nn.Linear(feature_dim, feature_dim)

    def forward(self, text_features, image_features):
        text_to_image = self.text_to_image(text_features)
        image_to_text = self.image_to_text(image_features)
        fused_text = text_features + image_to_text
        fused_image = image_features + text_to_image
        return fused_text, fused_image

class ProgressiveFusion(nn.Module):
    def __init__(self, feature_dim, num_stages=3):
        super(ProgressiveFusion, self).__init__()
        self.stages = nn.ModuleList([CrossModalResidualBlock(feature_dim) for _ in range(num_stages)])

    def forward(self, text_features, image_features):
        for stage in self.stages:
            text_features, image_features = stage(text_features, image_features)
        return text_features, image_features

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid=3, k=3, noise_scale=0.1, noise_scale_base=0.1,
                 base_fun=nn.SiLU(), symbolic_enabled=True, bias_trainable=True, grid_eps=1.0,
                 grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, device='cpu', seed=0):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid = grid
        self.k = k
        self.noise_scale = noise_scale
        self.noise_scale_base = noise_scale_base
        self.base_fun = base_fun
        self.symbolic_enabled = symbolic_enabled
        self.bias_trainable = bias_trainable
        self.grid_eps = grid_eps
        self.grid_range = grid_range
        self.sp_trainable = sp_trainable
        self.sb_trainable = sb_trainable
        self.device = device
        self.seed = seed

        torch.manual_seed(seed)
        self.fc = nn.Linear(in_features, out_features, bias=bias_trainable)
        self.activation = base_fun

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x

class KAN(nn.Module):
    def __init__(self, width=None, grid=3, k=3, noise_scale=0.1, noise_scale_base=0.1, base_fun=nn.SiLU(), symbolic_enabled=True,
                 bias_trainable=True, grid_eps=1.0, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True,
                 device='cpu', seed=0):
        super(KAN, self).__init__()
        self.width = width
        self.grid = grid
        self.k = k
        self.noise_scale = noise_scale
        self.noise_scale_base = noise_scale_base
        self.base_fun = base_fun
        self.symbolic_enabled = symbolic_enabled
        self.bias_trainable = bias_trainable
        self.grid_eps = grid_eps
        self.grid_range = grid_range
        self.sp_trainable = sp_trainable
        self.sb_trainable = sb_trainable
        self.device = device
        self.seed = seed

        torch.manual_seed(seed)
        self.layers = nn.ModuleList()
        for i in range(len(width) - 1):
            self.layers.append(KANLayer(in_features=width[i], out_features=width[i+1], grid=grid, k=k,
                                        noise_scale=noise_scale, noise_scale_base=noise_scale_base,
                                        base_fun=base_fun, symbolic_enabled=symbolic_enabled, bias_trainable=bias_trainable,
                                        grid_eps=grid_eps, grid_range=grid_range, sp_trainable=sp_trainable,
                                        sb_trainable=sb_trainable, device=device))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# from huggingface_hub import hf_hub_download
class CombinedModel(nn.Module):
    def __init__(self, kan_width=[1024, 512, 256, 7], kan_grid=3, kan_k=3, kan_noise_scale=0.1,
                 kan_noise_scale_base=0.1,
                 kan_base_fun=nn.SiLU(), kan_symbolic_enabled=True, kan_bias_trainable=True, kan_grid_eps=1.0,
                 kan_grid_range=[-1, 1], kan_sp_trainable=True, kan_sb_trainable=True, device='cpu', seed=0):
        super(CombinedModel, self).__init__()
        self.image_model = timm.create_model('swin_large_patch4_window7_224', pretrained=True,
                                             pretrained_cfg_overlay=dict(file=r'swin_large_patch4_window7_224.ms_in22k_ft_in1k\pytorch_model.bin'))  # 使用 Swin Transformer
        self.text_model = RobertaModel.from_pretrained("roberta-base")  # 使用 RoBERTa

        # 冻结预训练模型的参数
        for param in self.image_model.parameters():
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False

        self.image_model.head = nn.Identity()
        # 调整文本模型的输出维度
        self.text_fc = nn.Linear(self.text_model.config.hidden_size, 512)
        self.feature_reducer = nn.Linear(1536, 512)

        # 自适应特征模块
        self.adaptive_image_module = AdaptiveFeatureModule(512)
        self.adaptive_text_module = AdaptiveFeatureModule(512)
        # 使用KAN层作为融合模块
        # Initialize KAN model
        self.kan_model = KAN(width=kan_width, grid=kan_grid, k=kan_k, noise_scale=kan_noise_scale,
                             noise_scale_base=kan_noise_scale_base, base_fun=kan_base_fun,
                             symbolic_enabled=kan_symbolic_enabled, bias_trainable=kan_bias_trainable,
                             grid_eps=kan_grid_eps, grid_range=kan_grid_range, sp_trainable=kan_sp_trainable,
                             sb_trainable=kan_sb_trainable, device=device, seed=seed)

        # 最终分类层
        self.final_fc = nn.Linear(512, 7)
        # Progressive Fusion Model integration
        self.progressive_fusion = ProgressiveFusion(512)
        self.image_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, images, input_ids):
        image_features = self.image_model(images)
        image_features = self.image_model.avgpool(image_features.permute(0, 3, 1, 2))
        image_features = image_features.squeeze()
        image_features = self.feature_reducer(image_features)
        image_features_ad = self.adaptive_image_module(image_features)
        image_features = (image_features_ad + image_features) / 2

        text_outputs = self.text_model(input_ids=input_ids)
        text_features = text_outputs.pooler_output
        text_features = self.text_fc(text_features)
        text_features_ad = self.adaptive_text_module(text_features)
        text_features = (text_features_ad + text_features) / 2
        # Use progressive fusion to integrate text and image features
        fused_text, fused_image = self.progressive_fusion(text_features, image_features)
        combined_features = torch.cat((fused_text, fused_image), dim=1)
        # 最终分类
        outputs = self.kan_model(combined_features)
        return outputs, image_features, text_features

# 对比学习是一种自监督学习方法，其核心思想是通过比较正样本和负样本来学习数据的表示
# 对比损失（contrastive loss），以使正样本的相似度高于负样本的相似度
def itc_loss(image_features, text_features, negative_texts_features):
    # 计算图像和文本特征之间的余弦相似度
    normalized_images = F.normalize(image_features, p=2, dim=1)
    normalized_texts = F.normalize(text_features, p=2, dim=1)
    similarity = torch.mm(normalized_images, normalized_texts.t())

    # 简化版的损失函数，这里只计算正样本的相似度损失
    diagonal = torch.diagonal(similarity)
    positive_loss = 1 - diagonal.mean()  # 目标是使得对角线上的相似度尽可能接近1

    # 计算负样本的损失
    negative_loss = 0
    for neg_text_feat in negative_texts_features:
        neg_similarity = torch.mm(normalized_images, neg_text_feat.t())
        negative_loss += (1 - neg_similarity.diagonal()).mean()
    negative_loss /= len(negative_texts_features)

    # 总损失是正样本损失和负样本损失的和
    total_loss = positive_loss + negative_loss
    return total_loss

# 函数计算准确率
def calculate_accuracy(y_true, y_pred):
    predicted = y_pred.argmax(1)
    correct = (predicted == y_true).float()
    accuracy = correct.sum() / len(correct)
    return accuracy


def train(model, train_dataloader, val_dataloader, device, optimizer, optimizer_itc, itc_loss, loss_fn, epochs):
    model.train()
    best_val_loss = float('inf')  # 初始化最佳验证损失
    epochs_without_improvement = 0  # 连续没有改善的epoch数
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for images, input_ids, labels, negative_texts in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
            images, input_ids, labels = images.to(device), input_ids.to(device), labels.to(device)
            outputs, image_features, text_features = model(images, input_ids)

            optimizer.zero_grad()
            classification_loss = loss_fn(outputs, labels)
            classification_loss.backward(retain_graph=True)
            optimizer.step()

            total_loss += classification_loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            # 处理ITC损失
            optimizer_itc.zero_grad()
            # 获取负样本的文本特征
            negative_texts_features = []
            for neg_text in negative_texts:
                neg_text = neg_text.to(device)
                neg_text_outputs = model.text_model(input_ids=neg_text)
                neg_text_features = neg_text_outputs.pooler_output
                neg_text_features = model.text_fc(neg_text_features)
                negative_texts_features.append(neg_text_features)
            itc_loss_value = itc_loss(image_features, text_features, negative_texts_features)
            itc_loss_value.backward()
            optimizer_itc.step()

        train_loss = total_loss / len(train_dataloader)
        train_acc = correct_predictions / total_predictions

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        val_acc, val_loss = evaluate(model, val_dataloader, device, loss_fn, test=False)
        logging.info(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # 检查验证损失是否有改善
        if val_loss < best_val_loss:
            best_val_loss = val_loss  # 更新最佳验证损失
            epochs_without_improvement = 0  # 重置计数器
        else:
            epochs_without_improvement += 1  # 增加计数器

        # 如果连续两个epoch没有改善，则提前停止训练
        if epochs_without_improvement >= 2:
            print("Validation loss did not improve for two consecutive epochs. Stopping training.")
            break

def evaluate(model, data_loader, device, loss_fn, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for images, encoded_texts, labels, negative_texts in data_loader:
            images = images.to(device)
            encoded_texts = encoded_texts.to(device)
            labels = labels.to(device)

            outputs, image_features, text_features = model(images, encoded_texts)
            if not test:
                loss = loss_fn(outputs, labels)
                loss_total += loss.item()
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        return acc, labels_all, predict_all
    else:
        return acc, loss_total / len(data_loader)

def model_test(model, test_dataloader, device, loss_fn, class_names):
    # 测试评估
    acc, y_true, y_pred = evaluate(model, test_dataloader, device, loss_fn, test=True)

    # 计算性能指标
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # 日志记录
    logging.info("Test Evaluation")
    logging.info(f"Test Accuracy: {acc:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")

    # 打印到控制台
    print("Test Accuracy:", acc)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 混淆矩阵图片的保存路径
    picture_path = 'picture'
    if not os.path.exists(picture_path):
        os.makedirs(picture_path)

        # 绘制并保存混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    # 获取当前时间
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # 混淆矩阵图片的文件名
    cm_filename = f"picture/{model_name}_{current_time}_confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_filename)


def plot_confusion_matrix(cm, class_names, filename):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    all_diseases = ['health', 'early_blight', 'late_blight', 'Bacterial_spot', 'leaf_mold', 'Septoria_leaf_spot', 'Yellow_Leaf_Curl_Virus']
    # 创建疾病到整数标签的映射
    disease_to_label = {disease: i for i, disease in enumerate(all_diseases)}
    # 获取当前时间
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # 获取脚本文件名（模型名）
    model_name = os.path.basename(__file__).split('.')[0]

    # 创建log文件夹路径（如果不存在）
    log_path = 'log'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # 创建日志文件的完整路径
    log_filename = os.path.join(log_path, f"{model_name}.log")

    # 配置logging
    logging.basicConfig(filename=log_filename,
                        level=logging.INFO,
                        format='%(asctime)s:%(levelname)s: %(message)s',
                        filemode='a')  # 使用追加模式

    # 在日志文件中添加时间戳和空行
    logging.info(f"Run at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 接下来的日志记录会追加到文件中
    logging.info("Starting training...")

    train_image_dir = "./data/train/images"
    train_text_files = './data/train/train.xlsx'
    val_image_dir = "./data/val/images"
    val_text_files = './data/val/val.xlsx'
    test_image_dir = "./data/test/images"
    test_text_files = './data/test/test.xlsx'

    train_merged_df = read_excel_file_and_encode_labels(train_text_files, disease_to_label)
    val_merged_df = read_excel_file_and_encode_labels(val_text_files, disease_to_label)
    test_merged_df = read_excel_file_and_encode_labels(test_text_files, disease_to_label)

    # 定义transform
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), ])
    # 初始化tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("./roberta-base")
    # 初始化数据集对象
    train_dataset = ImageTextDataset(dataframe=train_merged_df, image_dir=train_image_dir, tokenizer=tokenizer, transform=transform)
    val_dataset = ImageTextDataset(dataframe=val_merged_df, image_dir=val_image_dir, tokenizer=tokenizer, transform=transform)
    test_dataset = ImageTextDataset(dataframe=test_merged_df, image_dir=test_image_dir, tokenizer=tokenizer, transform=transform)
    # 数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombinedModel().to(device)
    # 模型参数量
    total_params, trainable_params = count_parameters(model)
    logging.info(f"Model total parameters: {total_params}")
    logging.info(f"Model trainable parameters: {trainable_params}")

    # 提取AdaptiveFeatureModule的参数
    adaptive_image_params = list(model.adaptive_image_module.parameters())
    adaptive_text_params = list(model.adaptive_text_module.parameters())


    # 定义只更新这些参数的优化器
    optimizer_itc = SophiaG(adaptive_image_params + adaptive_text_params)

    main_params = []
    for name, param in model.named_parameters():
        if 'adaptive_image_module' not in name and 'adaptive_text_module' not in name:
            main_params.append(param)
    optimizer = SophiaG(main_params)
    loss_fn = nn.CrossEntropyLoss()
    # print("Before training:", model.adaptive_image_module.delta.data)
    train(model, train_dataloader, val_dataloader, device, optimizer, optimizer_itc, itc_loss, loss_fn, epochs=20)
    # print("After training:", model.adaptive_image_module.delta.data)
    model_test(model, test_dataloader, device, loss_fn, all_diseases)
    # 模型训练结束时间
    end_time = datetime.datetime.now()
    # 计算并记录总时间
    total_time = end_time - start_time
    logging.info(f"Total time taken: {total_time}")
    logging.info("\n\n\n")  # 添加三个空行
