import graphviz

# This code is to visualize the CNN architecture!

dot = graphviz.Digraph(comment='Custom CNN Architecture')

dot.node('I', 'Input Layer')

dot.node('C1', 'Conv Layer 1\n64 filters, 3x3, ReLU')
dot.node('P1', 'Max Pool 1\n2x2')
dot.node('C2', 'Conv Layer 2\n128 filters, 3x3, ReLU')
dot.node('P2', 'Max Pool 2\n2x2')
dot.node('C3', 'Conv Layer 3\n256 filters, 3x3, ReLU')
dot.node('P3', 'Max Pool 3\n2x2')
dot.node('C4', 'Conv Layer 4\n512 filters, 3x3, ReLU')
dot.node('P4', 'Max Pool 4\n2x2')

dot.node('F', 'Flatten')
dot.node('FC1', 'FC Layer 1\n1024 units, ReLU, Dropout 50%')
dot.node('FC2', 'FC Layer 2\nOutput Layer')

dot.edge('I', 'C1')
dot.edge('C1', 'P1')
dot.edge('P1', 'C2')
dot.edge('C2', 'P2')
dot.edge('P2', 'C3')
dot.edge('C3', 'P3')
dot.edge('P3', 'C4')
dot.edge('C4', 'P4')
dot.edge('P4', 'F')
dot.edge('F', 'FC1')
dot.edge('FC1', 'FC2')

graph_path = 'Custom_CNN_Architecture'
dot.render(graph_path, format='png')

graph_path