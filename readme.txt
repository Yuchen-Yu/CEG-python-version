# CEG 模拟与可视化工具包

该工具包包含两个核心程序：`ceg_p.py` 和 `ceg_output.py`，分别用于级联灭绝（Cascading Extinction on Graphs, CEG）的模拟计算与结果可视化分析，适用于生态网络等复杂系统的级联效应研究。


## 1. ceg_p.py：级联灭绝模拟程序

### 功能说明
`ceg_p.py` 是CEG模型的核心模拟程序，基于原始C++版本实现，用于模拟网络中物种的初级灭绝与级联灭绝过程。程序通过配置网络结构、灭绝参数等，生成包含扰动强度、灭绝数量、连接度等关键指标的模拟结果。


### 命令行使用方法
基本语法：
```bash
python ceg_p.py -no_nodes <节点数> -matrix_file <网络矩阵文件> -no_connections <链接总数> -connections_file <捕食链接文件> -no_affect_nodes <受影响节点数> -extinction_inc <灭绝增量> -high_diversity <最高灭绝多样性> -target_nodes <目标节点列表> -sims <模拟次数> -output_file <输出文件> -CPU <CPU编号>
```

示例：
```bash
python ceg_p.py -no_nodes 36 -matrix_file ms1.txt -no_connections 220 -connections_file link.txt -no_affect_nodes 4 -extinction_inc 1 -high_diversity 251 -target_nodes 0 1 2 3 -sims 100 -output_file result.txt -CPU 1
```

参数说明：
- `-no_nodes`：网络中的节点总数
- `-matrix_file`：网络矩阵文件（包含节点大小、猎物节点数等信息）
- `-no_connections`：网络中的捕食链接总数
- `-connections_file`：捕食链接文件（定义节点间的捕食关系）
- `-no_affect_nodes`：遭受初级灭绝的节点数量
- `-extinction_inc`：每次模拟的灭绝水平增量
- `-high_diversity`：目标节点的最高灭绝多样性阈值
- `-target_nodes`：遭受初级灭绝的目标节点编号列表（数量需与`-no_affect_nodes`一致）
- `-sims`：模拟重复次数
- `-output_file`：模拟结果输出文件（txt格式）
- `-CPU`：CPU编号（用于并行计算标识）


### 输出结果构成
输出文件（如`result.txt`）为空格分隔的文本格式，每行代表一次模拟迭代的结果，包含以下内容：
1. 模拟编号（`smp_sim_no`）
2. 总扰动强度（`total_pert`）
3. 初始连接度比例（`no_edges / L_max`）
4. 静态连接度（`static_connectance`）
5. 动态连接度（`dynamic_connectance`）
6. 每个节点的详细数据（按节点顺序）：
   - 初级灭绝水平（`extinction_level`）
   - 次级灭绝数量（`secondary_extinct`）
   - 总灭绝数量（`no_extinct`）
   - 次级灭绝百分比（`secondary_percent`）
   - 总灭绝百分比（`total_extinct_percent`）
7. 总体统计：
   - 扰动比例（`pert_ratio`）
   - 总灭绝数量（`total_extinct`）
   - 灭绝比例（`extinct_ratio`）


### 超算并行计算配置
由于模拟次数（`-sims`）较大时计算耗时较长，可通过超算的任务调度系统（如SLURM）实现并行计算：

1. **多任务并行**：将总模拟次数拆分到多个CPU核心，每个核心运行一个`ceg_p.py`实例，通过`-CPU`参数区分任务，示例SLURM脚本（`run_ceg.sh`）：
   ```bash
   #!/bin/bash
   #SBATCH --job-name=ceg_sim
   #SBATCH --partition=parallel
   #SBATCH --nodes=1
   #SBATCH --ntasks=4  # 4个并行任务
   #SBATCH --cpus-per-task=1

   for cpu in {1..4}; do
       python ceg_p.py -no_nodes 36 -matrix_file ms1.txt -no_connections 220 -connections_file link.txt -no_affect_nodes 4 -extinction_inc 1 -high_diversity 251 -target_nodes 0 1 2 3 -sims 25 -output_file result_${cpu}.txt -CPU $cpu &
   done
   wait
   ```
2. **结果合并**：并行任务完成后，可通过简单脚本合并所有输出文件（如`cat result_*.txt > all_results.txt`）。


## 2. ceg_output.py：结果可视化与后处理程序

### 功能说明
`ceg_output.py` 用于处理`ceg_p.py`生成的模拟结果，实现数据清洗、 changepoint（突变点）分析、特定阈值下的扰动值提取，并支持可视化绘图，辅助分析扰动与次级灭绝的关系。


### 命令行使用方法
基本语法：
```bash
python ceg_output.py <输入文件> [--output_prefix <输出前缀>] [--plot] [--segments <分段数>] [--bkps <每段突变点数>]
```

参数说明：
- `<输入文件>`：`ceg_p.py`生成的结果txt文件（必需）
- `--output_prefix`：输出文件前缀（默认：`ms1`）
- `--plot`：启用绘图功能（默认关闭）
- `--segments`：changepoint分析的分段数（默认：100）
- `--bkps`：每段的突变点数量（默认：10）


### 输出文件
程序会生成以下输出文件（前缀由`--output_prefix`指定）：
1. `<前缀>.csv`：清洗后的核心数据（扰动值与次级灭绝值）
2. `<前缀>cp_np.csv`：changepoint分析结果（各分段的突变点对应的扰动值）
3. `<前缀>selp.csv`：特定阈值（0.1, 0.2, 0.3, 0.4）下的最小扰动值
4. `<前缀>_plot.png`：扰动与次级灭绝的散点图（仅当`--plot`启用时生成）


## 工作流示例
1. 运行模拟：
   ```bash
   python ceg_p.py -no_nodes 36 -matrix_file ms1.txt -no_connections 220 -connections_file link.txt -no_affect_nodes 4 -extinction_inc 1 -high_diversity 251 -target_nodes 0 1 2 3 -sims 100 -output_file result.txt -CPU 1
   ```
2. 处理结果并绘图：
   ```bash
   python ceg_output.py result.txt --output_prefix ceg_result --plot --segments 200 --bkps 15
   ```
3. 查看输出：生成`ceg_result.csv`、`ceg_resultcp_np.csv`、`ceg_resultselp.csv`和`ceg_result_plot.png`。