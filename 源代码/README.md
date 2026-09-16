# 鸟类图像处理与生态评价

完整 CUB-200-2011 的传统 OpenCV 综合实践项目。包含 200 类、11,788 张图像的全量实验、九组分类对照、官方分割标注评价、AVONET 生态回归、PCA 与中文桌面界面。没有使用 SAM、YOLO 或深度预训练模型。

## 本机启动

双击 `启动界面.cmd`。已训练的模型无需重算。数据位于 `D:/图像处理综合实践/datasets`，配置文件为 `config.json`。

界面左侧可按英文名称或文件名检索，支持训练集与测试集筛选、每页 100 条浏览。选图后点击“开始分析”，或导入自己的图片。“批量处理”支持多张图片及 CSV 导出，“数据与实验统计”显示实际运行结果。

## 迁移安装

建议 Python 3.11，与本次实验环境一致。终端进入本目录后运行：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python download_data.py --root D:/bird_data
```

将 `config.json` 的 `data_root` 改为实际数据路径，然后：

```powershell
.venv\Scripts\python app.py
```

Tkinter 随标准 Windows Python 安装，不需要 pip 安装。Anaconda 环境使用现有 Python 即可。

## 全部实验复现

```powershell
python run_all.py
```

执行顺序为数据下载校验 → 物种映射 → 全量特征提取 → 训练评价 → 示例图 → 一致性校验。不要删除原始数据。若修改 `core.py`，请改用新的缓存版本目录，避免复用旧算法缓存。

`download_data.py` 独立运行时默认使用课程目录。`run_all.py` 自动将 config.json 中的数据路径传给下载脚本。下载支持断点续传，压缩包校验失败不会标为完成。

## 命令行批量推理

```powershell
python inference.py image1.jpg image2.png --csv results/my_batch.csv
```

损坏图像在输出中单独记录错误，其余图像继续处理。预测只使用像素，不读取图像名称中的类别信息。

## 数据与模型

- `models/production.joblib`：仅根据验证集选择的生产分类模型。
- `models/ecology.joblib`：独立的 full 特征生态回归模型。
- `models/pca.joblib`：训练集拟合的标准化器与 PCA。
- `results/features.npz`：全部 11,788 张图像的 5×795 维预处理特征及逐图分割分数。
- `results/image_index.csv`：特征矩阵的图像顺序，包含官方训练测试标记。
- `results/classification_metrics.json`：全部分类实验与选参记录。
- `results/test_predictions.csv`：全部官方测试图像的九组预测。
- `results/segmentation_per_image.csv`：全部图像逐图 IoU 和 Dice。
- `results/ecology_split.json`：按物种留出的独立生态评价协议。
- `results/verification.json`：自动验收结果。
- `metadata/species_crosswalk.csv`：200 类对应表，未确认字段明确留空。

## 评价解释

图像分类固定使用官方 5,994/5,794 训练测试划分。训练内部再分层划分验证集，测试集不参与参数选择。分割真值、边界框、人工部位与属性均不进入特征生成。PCA 与标准化器仅在训练集拟合。

AVONET 对应 195 个类别；Frigatebird、Nighthawk、Sayornis、Geococcyx、Tree Sparrow 五个含糊标签不猜测物种。它们全部参与图像实验，仅缺失生态标签。历史分类范围与人工明确别名在 match_note 中记录。

SVM 得分不是概率。AVONET 是物种平均性状，查表值以识别正确为前提。回归值是探索性估计，可能超出合理范围，不是照片中个体的实测值。二维投影对称性不等于生物学双侧对称性；PCA 图不证明适应性演化。

## 官方来源

- CUB 图像及课程研究使用限制：https://www.vision.caltech.edu/datasets/cub_200_2011/
- 图像归档：https://data.caltech.edu/records/65de6-vp158
- 分割标注：https://data.caltech.edu/records/w9d68-gec53
- AVONET 数据：https://api.figshare.com/v2/articles/16586228
- AVONET 论文：https://doi.org/10.1111/ele.13898
- eBird 名称对应：https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=csv&version=2021

保留实际 API 响应和哈希；eBird 版本参数可能被服务器忽略，历史物种范围按 AVONET 工作簿解释。

## GitHub 提交

本交付未上传远程账号。源码、指标、说明和截图可以作为仓库内容；模型与全量特征较大，建议通过 Release 或 Git LFS 提供。`.gitignore` 默认排除这些大文件，发布时必须另外附上它们，否则他人需要重新训练。原始 CUB 图像通过官方脚本获取并保留非商业研究及教育用途限制，不将数据来源误标为本人原创。

提交前补齐报告封面的姓名、学号和学院，并按老师要求打印报告、进行现场展示及提供实际仓库地址。
