import re
from collections import defaultdict
import random
import heapq
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class WordGraph:
    def __init__(self):
        self.graph = defaultdict(lambda: defaultdict(int))
        self.nx_graph = nx.DiGraph()
        self.sentences = []
        self.tf = defaultdict(dict)
        self.word_docs = defaultdict(int)

    def build_graph(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read().lower()
            text = re.sub(r'[^a-z\s]', ' ', text)
            self.sentences = text.splitlines()
            words = text.split()

            for i in range(len(words) - 1):
                word1, word2 = words[i], words[i + 1]
                self.graph[word1][word2] += 1

                if word1 not in self.tf:
                    self.tf[word1] = defaultdict(int)
                self.tf[word1][file_path] += 1
                if word2 not in self.tf:
                    self.tf[word2] = defaultdict(int)
                self.tf[word2][file_path] += 1

                if word1 not in self.word_docs:
                    self.word_docs[word1] = 1
                else:
                    self.word_docs[word1] += 1
                if word2 not in self.word_docs:
                    self.word_docs[word2] = 1
                else:
                    self.word_docs[word2] += 1

            for word1, neighbors in self.graph.items():
                for word2, weight in neighbors.items():
                    self.nx_graph.add_edge(word1, word2, weight=weight)

    def visualize_graph(self):
        if len(self.nx_graph.nodes) < 15:
            pos = nx.circular_layout(self.nx_graph)
        else:
            pos = nx.shell_layout(self.nx_graph)

        nx.draw(self.nx_graph, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=12,
                font_weight='bold',
                arrowsize=20, connectionstyle='arc3,rad=0.1')

        edge_labels = nx.get_edge_attributes(self.nx_graph, 'weight')
        for edge, weight in edge_labels.items():
            if weight > 1:
                nx.draw_networkx_edge_labels(self.nx_graph, pos, edge_labels={(edge[0], edge[1]): weight}, font_size=10)

        plt.title('Word Graph Visualization')
        plt.show()

    def find_bridge_words(self, word1, word2):
        word1, word2 = word1.lower(), word2.lower()
        if word1 not in self.graph or word2 not in self.graph:
            return f"No {word1} or {word2} in the graph!"

        bridge_words = []
        for neighbor in self.graph[word1]:
            if word2 in self.graph[neighbor]:
                bridge_words.append(neighbor)

        if not bridge_words:
            return f"No bridge words from {word1} to {word2}!"
        return f"The bridge words from {word1} to {word2} are: {', '.join(bridge_words)}"

    def generate_new_text(self, input_text):
        input_text = input_text.lower()
        input_text = re.sub(r'[^a-z\s]', ' ', input_text)
        words = input_text.split()

        if len(words) < 2:
            return input_text

        result = [words[0]]
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]
            bridge_words = self.find_bridge_words(word1, word2)
            if "bridge words" in bridge_words:
                bridge_word = random.choice(bridge_words.split(': ')[1].split(', '))
                result.append(bridge_word)
            result.append(word2)

        return ' '.join(result)

    def calculate_shortest_path(self, start_word, end_word=None):
        start_word = start_word.lower()
        if start_word not in self.graph:
            return f"Word {start_word} not in the graph!"

        distances = {node: float('inf') for node in self.graph}
        distances[start_word] = 0
        paths = {start_word: [start_word]}
        heap = [(0, start_word)]

        while heap:
            current_distance, current_node = heapq.heappop(heap)
            if current_distance > distances[current_node]:
                continue

            for neighbor, weight in self.graph[current_node].items():
                if neighbor not in distances:
                    continue
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    paths[neighbor] = paths[current_node] + [neighbor]
                    heapq.heappush(heap, (distance, neighbor))

        if end_word:
            end_word = end_word.lower()
            if end_word not in self.graph:
                return f"Word {end_word} not in the graph!"
            if distances[end_word] == float('inf'):
                return f"No path from {start_word} to {end_word}!"
            return f"最短路径: {' -> '.join(paths[end_word])}\n路径权重: {distances[end_word]}"
        else:
            results = []
            for node, distance in distances.items():
                if distance != float('inf') and node != start_word:
                    results.append(f"到 {node} (路径权重: {distance}):\n  最短路径: {' -> '.join(paths[node])}")
            return '\n\n'.join(results)

    def calculate_pagerank(self, d=0.85, custom_nodes=None):
        nodes = list(self.graph.keys())
        n = len(nodes)
        M = np.zeros((n, n))
        for i, node1 in enumerate(nodes):
            out_degree = sum(self.graph[node1].values())
            for j, node2 in enumerate(nodes):
                if node2 in self.graph[node1]:
                    M[i][j] = self.graph[node1][node2] / out_degree if out_degree > 0 else 0

        pr = np.ones(n) / n
        max_iterations = 100
        tolerance = 1e-6

        for _ in range(max_iterations):
            new_pr = d * np.dot(M, pr) + (1 - d) / n
            if np.linalg.norm(new_pr - pr) < tolerance:
                break
            pr = new_pr

        pr_dict = {nodes[i]: pr[i] for i in range(n)}
        if custom_nodes:
            result = []
            for node in custom_nodes:
                if node in pr_dict:
                    result.append(f"{node}'s PR value: {pr_dict[node]}")
                else:
                    result.append(f"Word {node} not in the graph!")
            return '\n'.join(result)
        return pr_dict

    def random_walk(self):
        if not self.graph:
            return "Graph is empty!"

        start_node = random.choice(list(self.graph.keys()))
        path = [start_node]
        visited_edges = set()

        while True:
            current_node = path[-1]
            if not self.graph[current_node]:
                break
            next_node = random.choice(list(self.graph[current_node].keys()))
            edge = (current_node, next_node)
            if edge in visited_edges:
                break
            path.append(next_node)
            visited_edges.add(edge)

        result = "Random walk result: Random walk completed: " + ' -> '.join(path)
        with open('random_walk.txt', 'w', encoding='utf-8') as file:
            file.write(result)
        return result


def main():
    print("欢迎使用文本分析程序！")
    file_path = input("请输入要分析的文本文件路径: ")

    graph = WordGraph()
    try:
        graph.build_graph(file_path)
        print(f"已成功从文件 {file_path} 构建图结构")
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return

    while True:
        print("\n请选择要执行的功能：")
        print("1. 可视化图结构")
        print("2. 查询桥接词")
        print("3. 基于桥接词生成新文本")
        print("4. 计算最短路径")
        print("5. 计算PageRank值")
        print("6. 进行随机游走")
        print("7. 退出程序")

        choice = input("请输入选项编号: ")

        if choice == "1":
            print("正在可视化图结构...")
            graph.visualize_graph()

        elif choice == "2":
            word1 = input("请输入第一个单词: ")
            word2 = input("请输入第二个单词: ")
            print(graph.find_bridge_words(word1, word2))

        elif choice == "3":
            input_text = input("请输入要扩展的文本: ")
            print(graph.generate_new_text(input_text))

        elif choice == "4":
            start_word = input("请输入起始单词: ")
            target_option = input("是否指定目标单词？(y/n): ").lower()
            if target_option == "y":
                end_word = input("请输入目标单词: ")
                print(graph.calculate_shortest_path(start_word, end_word))
            else:
                print(graph.calculate_shortest_path(start_word))

        elif choice == "5":
            d_option = input("是否自定义PageRank阻尼系数d？(y/n): ").lower()
            if d_option == "y":
                try:
                    d = float(input("请输入阻尼系数d (通常为0.85左右): "))
                    if not (0 < d < 1):
                        print("阻尼系数必须在0到1之间，使用默认值0.85")
                        d = 0.85
                except ValueError:
                    print("输入无效，使用默认值0.85")
                    d = 0.85
            else:
                d = 0.85

            custom_nodes_option = input("是否指定要计算PageRank值的单词？(y/n): ").lower()
            if custom_nodes_option == "y":
                nodes_input = input("请输入要计算PageRank值的单词，用逗号分隔: ")
                custom_nodes = [node.strip() for node in nodes_input.split(',')]
                print(graph.calculate_pagerank(d, custom_nodes))
            else:
                pr_dict = graph.calculate_pagerank(d)
                sorted_pr = sorted(pr_dict.items(), key=lambda x: x[1], reverse=True)
                print(f"所有节点的PageRank值（阻尼系数d={d}，按降序排列）:")
                for node, pr in sorted_pr:
                    print(f"{node}: {pr}")

        elif choice == "6":
            print(graph.random_walk())

        elif choice == "7":
            print("感谢使用文本分析程序，再见！")
            break

        else:
            print("无效的选项，请重新输入。")


if __name__ == "__main__":
    main()