# BioMoAX 代码说明

本目录对应当前论文的五项损失版本，包含模型、损失函数、训练入口、特征编码、检索评估、Ridge 基线和论文结果表。

这是待上传 GitHub 的本地整理版，尚未公开。当前论文对应的数值快照位于 `results/manuscript_20261024/`，`results/` 根目录保留历史导出表。新增的当前版本不包含构建产物、缓存、运行输出、原始图像或权重。请先阅读 `docs/UPLOAD_CHECKLIST.md`；正式仓库地址、权重与数据发布位置、软件许可证仍需作者确认。论文中尚未添加“代码已公开”的声明。

## 从哪里开始

1. 阅读英文 `README.md` 了解命令行接口。
2. 阅读 `docs/DATA.md`，准备符合格式的输入数据。
3. 阅读 `docs/REPRODUCTION.md`，了解论文训练与评估的对应关系。
4. 查看 `results/` 和 `configs/paper.json`，核对论文的结果与检查点。

```bash
python -m pip install -e .
python examples/synthetic_retrieval.py
biomoax evaluate --input runs/synthetic.npz --output runs/synthetic_metrics.json
python -m unittest discover -s tests -v
```

上面的例子使用人工构造数据，只用于检查接口是否可运行。

## 模型和训练

图像分支为五通道 ViT，表达分支为 MLP；两个分支分别具有药物级、MOA 级投影头。五项损失分别为药物配对、图像 MOA、表达 MOA、跨模态 MOA 和基因–药物图像配对。跨模态 MOA 项对表达表征停止梯度。CRISPR 到表达的额外损失在论文配置中关闭。

```bash
biomoax check-data --project-root /path/to/project_inputs
biomoax train --project-root /path/to/project_inputs --output runs/preflight --seed 43 --preflight
biomoax train --project-root /path/to/project_inputs --output runs/training --seed 43
```

训练入口保留原始归档训练流程，修正了机器绝对路径和输出路径。归档代码中的历史默认值由入口统一覆盖，不应直接运行 `legacy/` 内的单个脚本。预检需要完整的兼容数据和元数据，采用小缓存运行两次更新。正式训练要求支持 BF16 的 CUDA GPU，原始内存检查要求至少 100 GiB 主机内存。

模型代码从归档定义提取，来源及改动见 `SOURCE_MANIFEST.json`。`VERIFICATION.json` 记录发布包实际做过的验证。当前包未重新跑完 60 个 epoch，也未从原始图像重新生成论文全部图形。

## 使用范围

原始图像、处理后的大数组和预训练权重需单独提供。代码包不包含这些大文件。数据准备说明目前定义处理后接口和所需文件；从公共源完整重建所有输入的自动化下载、分割和匹配流程尚未作为一键功能发布。

主结果汇总三个检查点：43/29、45/28、46/28（随机种子/epoch）。网络局部图分别采用 45/28 和 42/28。具体连接表示检索关联。数据中的靶点注释不等同于完整生物学靶点集合。

Ridge 用训练配对估计标准化参数，分别拟合图像到表达和表达到图像两个回归，不使用 MOA 标签。检索通过显式正样本矩阵评估；同记录排除、跨细胞参考库和 CRISPR 正样本扩展由调用方根据元数据构建。

现有项目没有已确认的顶层许可证，公开发布前需确定代码授权；本包未擅自增加许可证，也未填写不存在的仓库或下载链接。
