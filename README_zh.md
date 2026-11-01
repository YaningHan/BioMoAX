# BioMoAX

BioMoAX 将细胞图像、L1000 表达谱及遗传扰动连接到共同的功能表征空间。

当前仓库包含训练与推理代码、岭回归基线、最新英文论文、结果表，以及独立的图像重绘与统计验证流程。

```bash
python -m pip install -e '.[reproduce]'
python -m unittest discover -s tests -v
python reproduce/run.py statistics
python reproduce/run.py figures
python reproduce/run.py paper
```

重绘使用随仓库提供的逐查询结果、固定 UMAP 坐标和小型图像示例；不需要训练数据或权重，也不代表重新训练或重新提取特征。输出位于 `reproduce/build/`。论文编译还需要 pdfLaTeX 和 latexmk。

完整说明见 [英文首页](README.md) 和 [复现说明](docs/REPRODUCTION.md)。权重校验信息已提供，下载地址尚未发布；原始训练图像、大型数组、缓存及第三方权重不放入 Git。
