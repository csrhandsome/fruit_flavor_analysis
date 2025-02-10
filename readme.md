# Python环境配置指南

## 准备工作

1. 安装VSCode
   - 从[官网](https://code.visualstudio.com/)下载并安装VSCode

2. 安装Python插件
   - 在VSCode扩展市场搜索"Python"
   - 安装Microsoft官方Python扩展

3. 安装Anaconda
   - 从[Anaconda官网](https://www.anaconda.com/download)下载并安装Anaconda
   - 安装完成后重启电脑

## 环境配置步骤

1. 在VSCode中配置Anaconda
   - 打开VSCode
   - 按`Ctrl+Shift+P`（Windows）或`Cmd+Shift+P`（Mac）
   - 输入"Python: Select Interpreter"
   - 选择Anaconda Python解释器

2. 创建项目环境
   ```bash
   conda create -f conda_environment.yaml
   ```

3. 激活环境
   ```bash
   conda activate fruit_flavor_analysis
   ```

## 注意事项

- 每次打开VSCode时，请确认右下角的Python解释器是否为`fruit_flavor_analysis`环境
- 如果不是，请按以下步骤切换：
  1. 点击VSCode右下角的Python解释器名称
  2. 从列表中选择`fruit_flavor_analysis`环境

## 运行项目

确认环境配置正确后，直接运行`main.py`文件即可启动项目。

## 常见问题

如果遇到环境相关的问题，请检查：
1. Anaconda是否正确安装
2. 环境是否正确激活
3. VSCode是否选择了正确的Python解释器