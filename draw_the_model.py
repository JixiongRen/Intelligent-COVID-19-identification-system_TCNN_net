import pygraphviz as pgv
import graphviz

graph = pgv.AGraph(directed=True)

# 添加节点
graph.add_node("Input")
graph.add_node("Convolutional Layer")
graph.add_node("Pooling Layer")
graph.add_node("Fully Connected Layer")
graph.add_node("Output")

# 添加边
graph.add_edge("Input", "Convolutional Layer")
graph.add_edge("Convolutional Layer", "Pooling Layer")
graph.add_edge("Pooling Layer", "Fully Connected Layer")
graph.add_edge("Fully Connected Layer", "Output")

# 设置节点属性
graph.get_node("Input").attr["shape"] = "box"
graph.get_node("Output").attr["shape"] = "box"
graph.get_node("Fully Connected Layer").attr["color"] = "red"

# 设置边属性
graph.get_edge("Convolutional Layer", "Pooling Layer").attr["color"] = "blue"
graph.get_edge("Pooling Layer", "Fully Connected Layer").attr["label"] = "Max Pooling"

# 渲染为PNG格式图像文件
graph.draw("model_structure.png", prog="dot", format="png")

