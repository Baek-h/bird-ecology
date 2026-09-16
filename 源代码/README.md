# 鸟类图像处理与生态评价

基于 OpenCV 传统图像处理的鸟类分析系统。使用完整 CUB-200-2011 数据集进行前景分割、200 类鸟类识别和图像特征分析，并结合 AVONET 物种性状开展生态回归与探索性聚类。

项目提供 Python Tkinter 桌面界面，支持图像浏览、名称检索、单张分析、批量处理及 CSV 导出。核心算法由传统图像处理算子和轻量机器学习模型组成，未使用 SAM、YOLO 或深度预训练模型。

## 功能与方法

| 模块 | 实现 |
| --- | --- |
| 图像预处理 | 等比例缩放、双边滤波 |
| 前景分割 | Lab 边界背景建模、Otsu 阈值、形态学处理、GrabCut |
| 几何对齐 | 前景主轴估计与仿射旋转 |
| 特征提取 | 颜色直方图、GLCM、LBP、形状、盒计数维数、投影对称性、HOG，共 795 维 |
| 分类与评价 | RBF-SVM、九组分类与消融实验、IoU / Dice 分割评价 |
| 生态分析 | AVONET 性状映射、岭回归、PCA 与 K-means 聚类 |
| 桌面界面 | 数据检索、单张与批量分析、结果导出、实验统计 |

## 实验结果

使用全部 **200 类、11,788 张图像**，按官方划分使用 5,994 张训练图像和 5,794 张测试图像。随机种子为 `2026`；分类参数和最终方案均根据训练集内部的验证集选择。

| 指标 | 结果 |
| --- | ---: |
| 所选分类方案 | without_hog |
| 测试集 Top-1 | 10.06% |
| 测试集 Top-5 | 26.56% |
| 测试集 Macro-F1 | 0.0987 |
| 完整分割流程平均 IoU | 0.4936 |
| 完整分割流程平均 Dice | 0.6102 |

传统特征在复杂背景下的细粒度识别仍存在明显局限。AVONET 匹配覆盖 195 个类别，其余 5 个标签因物种范围不明确而保留空值，全部图像仍参与分类和分割实验。

## 目录结构

全部程序集中在 `源代码/` 中：

```text
源代码/
├── app.py                    # Tkinter 界面
├── inference.py              # 单张与批量推理
├── core.py                   # 分割、对齐和特征提取
├── download_data.py          # 官方数据下载与校验
├── prepare_metadata.py       # AVONET 物种对应
├── extract_features.py       # 全量特征提取
├── train_evaluate.py         # 训练、消融与评价
├── select_production.py      # 按验证结果选择模型
├── cluster_analysis.py       # PCA 与聚类分析
├── make_examples.py          # 结果图与案例
├── verify.py                 # 数据与推理一致性校验
├── window_capture.py         # 应用窗口截图
├── run_all.py                # 完整实验入口
├── metadata/                 # 物种对应表和数据来源记录
├── config.json               # 数据路径与实验配置
├── requirements.txt          # 依赖版本
├── 启动界面.cmd
└── 重新运行实验.cmd
```

`models/` 和 `results/` 在运行实验后生成。本仓库当前已发布源码与元数据，**尚未发布包含模型及全量特征的 Release 附件**。仅下载仓库源码时，需要先运行完整实验，再启动界面。

## 安装与运行

已验证环境：Windows、Python 3.11.3、OpenCV 4.10.0、scikit-learn 1.6.1。以下命令在 **Windows CMD** 中逐行执行。

**1. 获取代码并安装依赖**

```bat
git clone https://github.com/Baek-h/bird-ecology.git
cd bird-ecology\源代码
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Python 安装需包含 Tcl/Tk，以运行 Tkinter 界面。

**2. 设置数据目录**

编辑 `config.json`，将 `data_root` 设置为本机可用路径，例如：

```json
{
  "data_root": "D:/bird_data",
  "seed": 2026,
  "workers": 4
}
```

**3. 下载数据并复现实验**

```bat
.venv\Scripts\python.exe run_all.py
```

程序依次执行官方下载与校验、物种对应、全量特征提取、分类及生态回归、生产模型选择、聚类分析、案例生成和一致性校验。首次运行会下载完整数据并训练模型，耗时取决于网络和计算机性能。

**4. 启动界面**

```bat
.venv\Scripts\python.exe app.py
```

已有完整 `models/`、`results/`、`metadata/` 和原始数据时，可跳过训练。确认 `config.json` 中的数据路径正确后启动界面。

界面中的“图像浏览与分析”用于检索和单张处理；“批量处理”支持多图分析与 CSV 导出；“数据与实验统计”展示数据规模和实验指标。

## 命令行批量分析

在模型与结果文件已准备好的情况下运行：

```bat
.venv\Scripts\python.exe inference.py image1.jpg image2.png --csv results/my_batch.csv
```

损坏图片单独记录错误，其余图片继续处理。预测依据图像像素，不使用文件名中的类别信息。

## 输出文件与可复现性

| 路径（相对于源代码目录） | 内容 |
| --- | --- |
| `models/production.joblib` | 按验证集选择的分类模型 |
| `models/ecology.joblib` | 生态性状回归模型 |
| `results/features.npz` | 全量预处理特征，形状为 11788×5×795 |
| `results/image_index.csv` | 特征对应的图像顺序与官方划分 |
| `results/feature_schema.json` | 特征字段定义 |
| `results/classification_metrics.json` | 分类指标与选参记录 |
| `results/test_predictions.csv` | 九组实验的测试集预测 |
| `results/segmentation_per_image.csv` | 逐图 IoU 与 Dice |
| `results/ecology_split.json` | 生态回归的物种留出划分 |
| `results/verification.json` | 自动校验记录 |

`features.npz` 保存的是预处理后提取的特征矩阵，不是处理后的图片文件集。原始数据保存在 `data_root` 指定目录，下载器支持断点续传并核对压缩包 MD5。

标准化与 PCA 仅在训练数据上拟合；分割真值只用于评分，不参与特征生成。生态回归采用整物种留出，避免同一物种的平均性状同时出现在训练和评价中。修改核心算法后，应更换特征缓存版本目录并重新提取特征。

## 结果解释

SVM 决策分数不是概率。界面分别显示识别后查询的 AVONET 物种平均性状和图像回归估计，两者均不等于照片中个体的实测值。二维投影对称性不能直接代表生物学双侧对称性，PCA 与聚类结果仅描述图像特征关联，不能证明适应性演化。

## 数据来源

- [CUB-200-2011 项目主页](https://www.vision.caltech.edu/datasets/cub_200_2011/)
- [CUB 图像归档](https://data.caltech.edu/records/65de6-vp158)
- [CUB 分割标注](https://data.caltech.edu/records/w9d68-gec53)
- [AVONET 数据记录](https://api.figshare.com/v2/articles/16586228)
- [AVONET 论文](https://doi.org/10.1111/ele.13898)
- [eBird 鸟类分类数据](https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=csv&version=2021)

数据权利归原作者或发布机构所有，使用时应遵守各数据源的许可与引用要求。仓库保留名称对应表、实际分类数据响应及来源校验记录；历史物种范围按 AVONET 工作簿解释。
