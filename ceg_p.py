#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CEG 1.5.0.0 - Complete Python Implementation
Cascading Extinction on Graphs
Converted from C++ maintaining original precision and algorithm logic
Original Copyright (C) 04/30/05 Peter Roopnarine
Python conversion maintains 6 significant figures precision
"""

import numpy as np
import sys
import time
import random
import copy
from dataclasses import dataclass, field
from typing import Set, List
import argparse  # 新增argparse库


@dataclass
class Species:
    """物种类 - 对应C++的SPECIES类"""
    name: int = 0
    tag: int = 0
    node: int = 0
    status: int = 1  # 1=存活, 0=灭绝
    no_prey: int = 0
    c_rank: int = 0
    K: float = 1.0
    link_strength: float = 0.0
    out_link_strength: float = 0.0
    in_link_strength: float = 0.0
    prey: Set[int] = field(default_factory=set)
    predators: Set[int] = field(default_factory=set)


@dataclass
class Node:
    """节点类 - 对应C++的NODE类"""
    name: int = 0
    size: int = 0
    linkage: int = 0
    no_prey: int = 0
    threshold: float = 0.0
    extinction_level: int = 0
    prey_nodes: Set[int] = field(default_factory=set)
    predatory_nodes: Set[int] = field(default_factory=set)


class CEGSimulator:
    """CEG模拟器主类"""

    def __init__(self, seed=None):
        """初始化随机数生成器"""
        if seed is None:
            seed = int(time.time())
        random.seed(seed)
        np.random.seed(seed)
        self.RAND_MAX = 2147483647

    def custom_rand(self):
        """模拟C语言的rand()函数"""
        return random.randint(0, self.RAND_MAX)

    def rand_float(self):
        """生成0到1之间的随机浮点数"""
        return self.custom_rand() / (self.RAND_MAX + 1.0)

    def read_matrix_file(self, filename, rows, cols):
        """读取矩阵文件"""
        with open(filename, 'r') as f:
            data = [float(x) for x in f.read().split()]

        matrix = []
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(data[i * cols + j])
            matrix.append(row)
        return matrix

    def read_connection_file(self, filename):
        """读取连接文件"""
        with open(filename, 'r') as f:
            data = [int(x) for x in f.read().split()]
        return data

    def calculate_prey_number(self, choice1, gamma, node_no_prey, rand_p):
        """
        计算捕食者数量
        choice1: 1=幂律, 2=指数, 3=混合, 4=均匀
        """
        node_no_prey_int = int(node_no_prey)

        if choice1 == 1:  # Power law distribution
            min_r = np.exp((gamma - 1) * np.log(node_no_prey) / gamma)
            if rand_p == 1.0:
                temp_p = self.rand_float()
                no_prey = 1 if temp_p == 0 else int(min_r * temp_p)
            else:
                max_prey = int(np.power(
                    np.power(node_no_prey, gamma - 1.0) / rand_p,
                    1.0 / gamma
                ))
                temp_p = self.rand_float()
                no_prey = 1 if temp_p == 0 else int(max_prey * temp_p)

            if no_prey > node_no_prey_int:
                no_prey = node_no_prey_int

        elif choice1 == 2:  # Exponential distribution
            min_r = np.log(node_no_prey)
            if rand_p == 1.0:
                no_prey = int(min_r)
            else:
                no_prey = int(np.log(node_no_prey) - np.log(rand_p))

            if no_prey > node_no_prey_int:
                no_prey = node_no_prey_int

        elif choice1 == 3:  # Mixed distribution
            if rand_p == 0:
                no_prey = node_no_prey_int
            else:
                no_prey = int(-1 * np.power(node_no_prey, 1 - (1 / gamma)) * np.log(rand_p))

            if no_prey == 0:
                no_prey = 1
            if no_prey > node_no_prey_int:
                no_prey = node_no_prey_int

        elif choice1 == 4:  # Uniform distribution
            no_prey = (self.custom_rand() % node_no_prey_int) + 1

        return no_prey

    def ceg_cascade(self, ceg_inc, ceg_no_nodes, ceg_no_affect_nodes,
                    ceg_total_species, ceg_extinction_vector1b,
                    extinction_vector2b, ceg_extinction_nodes_diversity,
                    ceg_Species2, ceg_Nodes2):
        """
        CEG级联灭绝核心算法 - 完整实现ceg.hpp
        """
        primary_extinction = 0

        # 初级灭绝：淘汰目标节点中的物种
        for count8 in range(ceg_no_nodes):
            for count9 in range(ceg_no_affect_nodes):
                if count8 == ceg_extinction_vector1b[count9]:
                    # 列出节点中的物种
                    node_species = []
                    for count10 in range(ceg_total_species):
                        b = ceg_Species2[count10].node
                        if b == count8:
                            node_species.append(ceg_Species2[count10].name)

                    # 随机选择物种进行灭绝
                    random.shuffle(node_species)

                    # 增加节点灭绝水平
                    temp_extinction_level = (ceg_Nodes2[ceg_extinction_vector1b[count9]].extinction_level +
                                             ceg_inc)

                    # 检查是否达到最大灭绝
                    if temp_extinction_level >= ceg_Nodes2[ceg_extinction_vector1b[count9]].size:
                        temp_extinction_level = ceg_Nodes2[ceg_extinction_vector1b[count9]].size

                    ceg_Nodes2[ceg_extinction_vector1b[count9]].extinction_level = temp_extinction_level

                    # 使适当数量的物种灭绝
                    for count11 in range(temp_extinction_level):
                        for count12 in range(ceg_total_species):
                            if (ceg_Species2[count12].name == node_species[count11] and
                                    ceg_Species2[count12].node == ceg_extinction_vector1b[count9]):
                                ceg_Species2[count12].status = 0
                                primary_extinction += 1

        # 级联灭绝循环
        no_changes = 1
        round_num = 1
        tally4 = 0

        while no_changes != 0:
            no_changes = 1
            round_num = round_num * -1

            for count13 in range(ceg_no_nodes):
                # 检查节点是否为捕食者
                if ceg_Nodes2[count13].no_prey > 0:
                    temp_threshold = ceg_Nodes2[count13].threshold

                    # 检查节点中的物种
                    for count14 in range(ceg_total_species):
                        # 防止复活检查
                        if ceg_Species2[count14].status == 0:
                            ceg_Species2[count14].link_strength = 0
                            ceg_Species2[count14].K = 0
                            continue

                        temp_out_link_s = 0.0
                        temp_in_link_s = 1.0

                        if (ceg_Species2[count14].node == ceg_Nodes2[count13].name and
                                ceg_Species2[count14].status == 1):
                            tally4 += 1

                            # 获取物种的猎物集合
                            c = ceg_Species2[count14].prey.copy()
                            temp_no_prey = float(len(c))

                            # 检查猎物的灭绝状态
                            to_remove = set()
                            for prey_idx in c:
                                if ceg_Species2[prey_idx].status == 0:
                                    to_remove.add(prey_idx)
                            c -= to_remove

                            # 重新计算链接强度
                            if len(c) > 0:
                                ceg_Species2[count14].link_strength = 1.0 / len(c)
                            else:
                                ceg_Species2[count14].status = 0
                                ceg_Species2[count14].link_strength = 0

                            # 更新消费收入
                            for prey_idx in c:
                                if ceg_Species2[prey_idx].status == 1:
                                    temp_in_link_s += ((ceg_Species2[prey_idx].in_link_strength -
                                                        ceg_Species2[prey_idx].out_link_strength) *
                                                       ceg_Species2[count14].link_strength)
                                    if temp_in_link_s == 1:
                                        ceg_Species2[count14].status = 0

                            # 基于猎物可用性更新消费者状态
                            if len(c) == 0:
                                ceg_Species2[count14].status = 0
                                no_changes += 1

                            # 检查唯一猎物是否为自己
                            if len(c) == 1:
                                for prey_idx in c:
                                    if (ceg_Species2[prey_idx].name == ceg_Species2[count14].name and
                                            ceg_Species2[prey_idx].node == ceg_Species2[count14].node):
                                        ceg_Species2[count14].status = 0
                                        no_changes += 1

                            # 获取捕食者集合
                            predators4 = ceg_Species2[count14].predators.copy()

                            # 检查捕食者状态并更新集合
                            to_remove = set()
                            for pred_idx in predators4:
                                if ceg_Species2[pred_idx].status == 0:
                                    to_remove.add(pred_idx)
                                elif ceg_Species2[pred_idx].status == 1:
                                    temp_out_link_s += ((ceg_Species2[pred_idx].in_link_strength -
                                                         ceg_Species2[pred_idx].out_link_strength) *
                                                        ceg_Species2[pred_idx].link_strength)
                            predators4 -= to_remove

                            # 更新捕食者列表
                            ceg_Species2[count14].predators = predators4

                            # 更新K并检查灭绝状态
                            new_K = temp_in_link_s - temp_out_link_s
                            delta_K = new_K - ceg_Species2[count14].K

                            if delta_K < 0:
                                if ceg_Species2[count14].K > 0 and ceg_Species2[count14].status == 1:
                                    extinction_pt = ceg_Species2[count14].K * temp_threshold
                                    if new_K < extinction_pt:
                                        ceg_Species2[count14].status = 0
                                        ceg_Species2[count14].link_strength = 0
                                        no_changes += 1

                                elif ceg_Species2[count14].K < 0 and ceg_Species2[count14].status == 1:
                                    extinction_pt = ceg_Species2[count14].K * (2 - temp_threshold)
                                    if new_K < extinction_pt:
                                        ceg_Species2[count14].status = 0
                                        ceg_Species2[count14].link_strength = 0
                                        no_changes += 1

                                elif ceg_Species2[count14].K == 0 and ceg_Species2[count14].status == 1:
                                    temp_K = 1.0
                                    extinction_pt = temp_K * temp_threshold
                                    if (1 + new_K) < extinction_pt:
                                        ceg_Species2[count14].status = 0
                                        ceg_Species2[count14].link_strength = 0
                                        no_changes += 1

                            ceg_Species2[count14].in_link_strength = temp_in_link_s
                            ceg_Species2[count14].out_link_strength = temp_out_link_s

                # 检查节点是否为初级生产者并更新
                if ceg_Nodes2[count13].no_prey == 0:
                    temp_threshold2 = ceg_Nodes2[count13].threshold

                    for count25 in range(ceg_total_species):
                        # 防止复活检查
                        if ceg_Species2[count25].status == 0:
                            ceg_Species2[count25].link_strength = 0
                            ceg_Species2[count25].K = 0
                            continue

                        temp_out_link_s = 0.0

                        if (ceg_Species2[count25].node == ceg_Nodes2[count13].name and
                                ceg_Species2[count25].status == 1):
                            # 获取捕食者
                            predators3 = ceg_Species2[count25].predators.copy()

                            # 检查捕食者状态并更新集合
                            to_remove = set()
                            for pred_idx in predators3:
                                if ceg_Species2[pred_idx].status == 0:
                                    to_remove.add(pred_idx)
                                elif ceg_Species2[pred_idx].status == 1:
                                    temp_out_link_s += ((ceg_Species2[pred_idx].in_link_strength -
                                                         ceg_Species2[pred_idx].out_link_strength) *
                                                        ceg_Species2[pred_idx].link_strength)
                            predators3 -= to_remove

                            # 更新捕食者列表
                            ceg_Species2[count25].predators = predators3

                            new_K = 0 - temp_out_link_s
                            delta_K = new_K - ceg_Species2[count25].K

                            if delta_K < 0 and ceg_Species2[count25].K != 0:
                                if ceg_Species2[count25].K < 0:
                                    extinction_pt = ceg_Species2[count25].K * (2 - temp_threshold2)
                                    if new_K < extinction_pt:
                                        ceg_Species2[count25].status = 0
                                        no_changes += 1

                if no_changes == 1:
                    no_changes = 0

        return ceg_Nodes2

    def run_simulation(self, args):
        """运行完整模拟"""
        print("\nCEG 1.0")
        print("Copyright (C) 04/30/05 Peter Roopnarine")
        print("CEG (Cascading Extinction on Graphs) - Python Implementation")
        print("=" * 70)

        # 从解析后的参数获取值
        no_nodes = args.no_nodes
        matrix_file = args.matrix_file
        no_connections = args.no_connections
        connections_file = args.connections_file
        no_affect_nodes = args.no_affect_nodes
        extinction_inc = args.extinction_inc
        high_diversity = args.high_diversity
        target_nodes = args.target_nodes
        sims = args.sims
        output_file = args.output_file
        CPU = args.CPU

        # 验证目标节点数量是否与no_affect_nodes一致
        if len(target_nodes) != no_affect_nodes:
            print(f"错误: 目标节点数量({len(target_nodes)})与指定的受影响节点数({no_affect_nodes})不匹配")
            sys.exit(1)

        print(f"\n请提供网络参数...")
        print(f"\n网络节点数 = {no_nodes}")

        # 读取网络矩阵
        network_matrix = self.read_matrix_file(matrix_file, no_nodes, 5)
        print(f"\n网络矩阵文件名: {matrix_file}")
        print()
        for row in network_matrix:
            print(" ".join(f"{x:.6g}" for x in row))

        # 计算总消费者多样性
        total_richness = sum(row[0] for row in network_matrix if row[1] > 0)

        # 读取捕食连接
        print(f"\n总链接数 = {no_connections}")
        connections = self.read_connection_file(connections_file)
        print(f"\n捕食链接文件名: {connections_file}")

        # 构建节点
        print("\n好的，让我们构建节点")
        nodes = []
        total_species = 0
        link_tally = 0

        for i in range(no_nodes):
            node = Node()
            node.name = i
            node.size = int(network_matrix[i][0])

            print(f"\n节点 {i} 中的物种数: {node.size:.6g}")

            input_no_prey = int(network_matrix[i][1])
            print(f"猎物节点数: {input_no_prey:.6g}")

            if input_no_prey == 0:
                node.threshold = network_matrix[i][4]
                print(f"灭绝阈值: {node.threshold:.6g}")

            if input_no_prey > 0:
                prey_nodes = set()
                for j in range(input_no_prey):
                    prey_node = connections[link_tally]
                    print(f"猎物节点编号 {j + 1}: {prey_node:.6g}")
                    prey_nodes.add(prey_node)
                    link_tally += 1

                node.prey_nodes = prey_nodes

                choice1 = int(network_matrix[i][2])
                print(f"节点的入度链接分布, (1) 幂律, (2) 指数, (3) 混合: {choice1:.6g}")

                if choice1 in [1, 3]:
                    gamma = network_matrix[i][3]
                    print(f"Gamma值: {gamma:.6g}")

                node.linkage = choice1
                node.threshold = network_matrix[i][4]
                print(f"灭绝阈值: {node.threshold:.6g}")

            nodes.append(node)
            total_species += node.size
            print()

        # 计算每个节点的可用猎物物种数
        print("\n根据链接分布计算每个物种的猎物数...")
        for node in nodes:
            if len(node.prey_nodes) > 0:
                no_prey_species = sum(nodes[prey_idx].size
                                      for prey_idx in node.prey_nodes)
                node.no_prey = no_prey_species

        # 获取每个节点的捕食者节点
        for i, node in enumerate(nodes):
            print(f"\n节点编号 {i}, ", end="")
            predatory_nodes = set()
            for j in range(no_nodes):
                if i in nodes[j].prey_nodes:
                    predatory_nodes.add(j)
            node.predatory_nodes = predatory_nodes
            print(f"捕食者节点: {predatory_nodes}")
            node.extinction_level = 0

        print("\n完成")

        # 设置灭绝参数
        print("\n让我们设置灭绝参数...")
        print(f"\n遭受初级灭绝的节点数 = {no_affect_nodes}")
        print(f"\n灭绝水平增量 = {extinction_inc}")
        print(f"\n最高灭绝目标节点多样性 = {high_diversity}")

        extinction_reps = int(np.floor(high_diversity / extinction_inc))

        extinction_matrix2 = []
        extinction_vector1 = []
        extinction_vector2 = []

        print("\n现在让我们选择灭绝目标节点...")
        for i in range(no_affect_nodes):
            target_node = target_nodes[i]
            print(f"\n节点编号 {i + 1} = {target_node}")
            extinction_matrix2.append([target_node, extinction_inc])
            extinction_vector1.append(target_node)
            extinction_vector2.append(extinction_inc)

        print(f"\n模拟次数 = {sims}")

        # 计算灭绝节点的多样性总和
        extinction_nodes_diversity = 0
        for count19 in range(no_nodes):
            for count20 in range(no_affect_nodes):
                if nodes[count19].name == extinction_matrix2[count20][0]:
                    extinction_nodes_diversity += nodes[count19].size
        print(f"\n灭绝多样性 = {extinction_nodes_diversity}")

        # 计算最大网络连接度
        L_max = total_species * (total_species - 1) / 2.0

        # 运行模拟
        print(f"在CPU {CPU}上开始模拟...")
        start_time = time.process_time()

        with open(output_file, 'w') as out1:
            for count13 in range(sims):
                print(f"\n在CPU {CPU}上运行模拟 {count13 + 1}")
                sim_no = count13

                # 设置初始边数
                no_edges = 0.0

                # 重置节点灭绝为0
                for count25 in range(no_nodes):
                    nodes[count25].extinction_level = 0

                # 构建每个物种并分配到节点
                species_list = []
                tally1 = 0

                for count3 in range(no_nodes):
                    # 获取猎物节点
                    prey_nodes = nodes[count3].prey_nodes
                    no_prey_nodes = len(prey_nodes)

                    # 获取节点大小
                    node_size = nodes[count3].size

                    # 设置竞争等级
                    temp_rankings = list(range(1, node_size + 1))
                    random.shuffle(temp_rankings)

                    for count4 in range(node_size):
                        species = Species()
                        species.name = count4
                        species.node = count3
                        species.status = 1
                        species.K = 1.0
                        species.c_rank = temp_rankings[count4]

                        if no_prey_nodes == 0:
                            species_list.append(species)
                            tally1 += 1
                            continue

                        # 设置猎物数量
                        if no_prey_nodes > 0:
                            node_no_prey = float(nodes[count3].no_prey)

                            # 计算该物种的猎物数量
                            rand_p = self.rand_float()
                            choice1 = nodes[count3].linkage
                            gamma = network_matrix[count3][3] if choice1 in [1, 3] else 0

                            no_prey = self.calculate_prey_number(
                                choice1, gamma, node_no_prey, rand_p
                            )

                            species.no_prey = no_prey
                            no_edges += no_prey
                            species.link_strength = 1.0 / no_prey

                        species_list.append(species)
                        tally1 += 1

                # 为每个物种生成网络链接
                tally2 = 0

                # 生成入链接
                for count5 in range(no_nodes):
                    node_size = nodes[count5].size
                    prey_nodes = nodes[count5].prey_nodes
                    no_prey_nodes = len(prey_nodes)

                    if no_prey_nodes == 0:
                        tally2 += node_size
                        continue

                    # 构建所有潜在猎物物种的向量
                    prey_species = []
                    for count7 in range(total_species):
                        if species_list[count7].node in prey_nodes:
                            prey_species.append(count7)

                    # 将猎物分配给消费者物种
                    for count6 in range(node_size):
                        its_no_prey = species_list[tally2].no_prey

                        # 随机化猎物顺序
                        random.shuffle(prey_species)

                        # 选择适当数量的猎物并分配到集合
                        prey_set = set()
                        for count7 in range(its_no_prey):
                            prey_set.add(prey_species[count7])

                        species_list[tally2].prey = prey_set
                        tally2 += 1

                # 生成出链接
                for count19 in range(total_species):
                    home_node = species_list[count19].node

                    # 获取我们节点的捕食者
                    predatory_nodes2 = set()
                    for count20 in range(no_nodes):
                        if nodes[count20].name == home_node:
                            predatory_nodes2 = nodes[count20].predatory_nodes

                    # 获取我们物种的捕食者
                    predators1 = set()
                    for count23 in range(total_species):
                        predators_node = species_list[count23].node

                        # 检查该物种是否为我们节点的捕食者
                        if predators_node in predatory_nodes2:
                            # 检查我们的物种是否为该物种的猎物
                            predators_prey1 = species_list[count23].prey
                            if count19 in predators_prey1:
                                predators1.add(count23)

                    species_list[count19].predators = predators1

                # 计算初始出链接强度和消费收入
                for count24 in range(total_species):
                    out_link_strength = 0.0
                    predators2 = species_list[count24].predators
                    for temp_iterator3 in predators2:
                        out_link_strength += (species_list[temp_iterator3].K *
                                              species_list[temp_iterator3].link_strength)
                    species_list[count24].out_link_strength = out_link_strength

                    in_link_strength = 0.0
                    prey3 = species_list[count24].prey
                    for temp_iterator10 in prey3:
                        in_link_strength += (species_list[count24].link_strength *
                                             species_list[temp_iterator10].K)
                    species_list[count24].in_link_strength = in_link_strength

                # 重新计算K
                for count41 in range(total_species):
                    species_list[count41].K = (species_list[count41].in_link_strength -
                                               species_list[count41].out_link_strength)

                # 复制灭绝矩阵
                extinction_matrix = copy.deepcopy(extinction_matrix2)
                extinction_vector1b = extinction_vector1.copy()
                extinction_vector2b = extinction_vector2.copy()

                # 复制节点数组
                nodes2 = copy.deepcopy(nodes)

                # 开始该网络的级联系列
                for count14 in range(extinction_reps):
                    # 复制物种数组
                    species2 = copy.deepcopy(species_list)

                    # 开始ceg级联
                    output_nodes = self.ceg_cascade(
                        extinction_inc, no_nodes, no_affect_nodes,
                        total_species, extinction_vector1b,
                        extinction_vector2b, extinction_nodes_diversity,
                        species2, nodes2
                    )

                    # 增加下一轮的灭绝水平
                    total_planned_extinction = 0
                    for count15 in range(no_affect_nodes):
                        total_planned_extinction += extinction_matrix[count15][1]
                        extinction_matrix[count15][1] += extinction_inc
                        extinction_vector2b[count15] = extinction_matrix[count15][1]

                    # 汇总本轮结果
                    # 计算网络的连接度度量
                    c_no_species = 0.0
                    no_links = 0.0

                    for count29 in range(total_species):
                        if species2[count29].status == 1:
                            c_no_species += 1

                        if species2[count29].no_prey > 0 and species2[count29].status == 1:
                            temp_prey_set1 = species2[count29].prey
                            for temp_prey_idx in temp_prey_set1:
                                if species2[temp_prey_idx].status == 1:
                                    no_links += 1

                    L_expected = c_no_species * (c_no_species - 1) / 2.0

                    if c_no_species >= 2:
                        static_connectance = no_links / L_max
                        dynamic_connectance = no_links / L_expected
                    else:
                        static_connectance = 0.0
                        dynamic_connectance = 0.0

                    # 计算总体扰动
                    total_pert = 0
                    for count45 in range(no_nodes):
                        total_pert += nodes2[count45].extinction_level

                    # 计算SMP并行模拟编号
                    if CPU <= 3:
                        smp_sim_no = sims * (CPU - 1) + count13
                    else:
                        smp_sim_no = 3 * (sims - 2) + count13

                    # 写入连接度数据
                    out1.write(f"{smp_sim_no} {total_pert} {no_edges / L_max:.6g} "
                               f"{static_connectance:.6g} {dynamic_connectance:.6g} ")

                    # 计算每个节点的灭绝情况
                    total_extinct = 0

                    for count16 in range(no_nodes):
                        no_extinct = 0
                        for count17 in range(total_species):
                            if (species2[count17].node == count16 and
                                    species2[count17].status == 0):
                                no_extinct += 1

                        secondary_extinct = no_extinct - nodes2[count16].extinction_level
                        node_size_remaining = nodes2[count16].size - nodes2[count16].extinction_level

                        # 计算次级灭绝百分比
                        if node_size_remaining > 0:
                            secondary_percent = (secondary_extinct / node_size_remaining) * 100
                        else:
                            secondary_percent = 0.0

                        if nodes2[count16].size > 0:
                            total_extinct_percent = (no_extinct / nodes2[count16].size) * 100
                        else:
                            total_extinct_percent = 0.0  # 节点大小为0时的默认值

                        out1.write(f"{nodes2[count16].extinction_level} "
                                   f"{secondary_extinct} "
                                   f"{no_extinct} "
                                   f"{secondary_percent:.6g} "
                                   f"{total_extinct_percent:.6g} ")

                        if nodes[count16].no_prey > 0:
                            total_extinct += no_extinct

                    # 写入总体统计
                    pert_ratio = total_pert / extinction_nodes_diversity
                    extinct_ratio = total_extinct / total_richness

                    out1.write(f"{pert_ratio:.6g} {total_extinct} {extinct_ratio:.6g}\n")

        elapsed_time = time.process_time() - start_time
        print(f"\nCPU {CPU}上的分析完成.")
        print(f"分析耗时 {elapsed_time:.6g} 秒的处理器时间.\n")

        return 0


def main():
    """主函数 - 使用argparse处理命令行参数"""
    parser = argparse.ArgumentParser(description='CEG (Cascading Extinction on Graphs) 模拟程序')

    # 定义命令行参数
    parser.add_argument('-no_nodes', type=int, required=True,
                        help='网络节点数')
    parser.add_argument('-matrix_file', type=str, required=True,
                        help='网络矩阵文件名')
    parser.add_argument('-no_connections', type=int, required=True,
                        help='链接总数')
    parser.add_argument('-connections_file', type=str, required=True,
                        help='捕食链接文件名')
    parser.add_argument('-no_affect_nodes', type=int, required=True,
                        help='遭受初级灭绝的节点数')
    parser.add_argument('-extinction_inc', type=int, required=True,
                        help='灭绝水平增量')
    parser.add_argument('-high_diversity', type=int, required=True,
                        help='最高灭绝目标节点多样性')
    parser.add_argument('-target_nodes', type=int, nargs='+', required=True,
                        help='目标节点编号列表(数量与no_affect_nodes一致)')
    parser.add_argument('-sims', type=int, required=True,
                        help='模拟次数')
    parser.add_argument('-output_file', type=str, required=True,
                        help='输出文件名')
    parser.add_argument('-CPU', type=int, required=True,
                        help='CPU编号')

    # 解析参数
    args = parser.parse_args()

    simulator = CEGSimulator()
    result = simulator.run_simulation(args)
    sys.exit(result)


if __name__ == "__main__":
    main()