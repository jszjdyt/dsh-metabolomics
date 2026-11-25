# 安装

## 依赖

### Python：专用生物信息环境

```bash
uv venv --python 3.12 ~/.dsh/runtimes/bio312
uv pip install --python ~/.dsh/runtimes/bio312/bin/python \
  "numpy<2.4" "pandas<3" scipy scikit-learn matplotlib seaborn anndata \
  scanpy squidpy umap-learn leidenalg igraph harmonypy scikit-misc \
  statsmodels openpyxl h5py networkx lifelines pillow certifi \
  matchms pyopenms
```

### R：ropls（PLS-DA / OPLS-DA 真包）

`bioconductor.org` 不可达时（本机情况）用源码装：

```bash
git clone --depth 1 https://git.bioconductor.org/packages/ropls /tmp/ropls_src
# 只去掉 MultiAssayExperiment / MultiDataSet 两个对象互操作依赖后安装：
R CMD INSTALL --no-test-load /tmp/ropls_build
Rscript -e 'requireNamespace("ropls", quietly=TRUE)'
```

（本机同时装了 `spacexr`（RCTD）供空间插件反卷积用。）

### 回退环境（宿主自带 python3）

```bash

```bash
python3 -c "import numpy,pandas,scipy,sklearn,matplotlib,statsmodels,openpyxl,h5py,anndata,lifelines,PIL,certifi"
```

缺包时：`python3 -m pip install -r requirements.txt`（不强制，插件自带降级）。

## 安装插件

方式一（推荐）：Harness → 设置 → 插件 → 添加本地插件 → 选择本目录。
方式二：让 Agent 调 `plugin_manager` `action=install_bundle, target=<本目录绝对路径>`。
方式三（高级）：在 `~/.dsh/profiles/<profile>/package.json` 的 `dependencies` 加
`"<包名>": "link:/本目录绝对路径"`，并在 `dsh.profile.bundles` 追加包名，然后重载宿主。

装好后重载页面：侧边栏出现工作台入口，对话里出现对应 Agent 工具。

## 常见问题

**Q：引擎徽章显示「Python 未就绪」？**
点徽章看原因。可在插件配置里指定 `python`（绝对路径），或设环境变量（spatial: `DSH_SPATIAL_PYTHON`；metabolomics: `DSH_METABOLOMICS_PYTHON`）。日志：`<临时目录>/<包名>/engine.log`。

**Q：GEO 检索慢/失败？**
需要能访问 NCBI。失败会给中文原因；已下载的文件可直接用「导入数据」打开。

**Q：数据/图存在哪？**
工作目录默认系统临时目录下 `<包名>/`（`analyses/` 图、`exports/` 导出、`downloads/` 下载）。可在插件配置 `workdir` 固定。

**Q：图和表能导出吗？**
「记录」tab 有一键导出（CSV/JSON/Methods/Markdown 报告/可复现脚本），也可以用 Agent 工具 `export`。
