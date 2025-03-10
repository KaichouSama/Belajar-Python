import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('C:/Users/zeroo/OneDrive/Documents/Project Mandiri/AI (Python)/Tugas 2/greedy.png')
plt.imshow(img)
plt.show()

graph = {
    'S': [('A', 1), ('B', 4)],
    'A': [('B', 2), ('C', 5), ('G', 12)],
    'B': [('C', 2)],
    'C': [('G', 3)]
}

H_table = {
    'S': 7,
    'A': 6,
    'B': 4,
    'C': 2,
    'G': 0
}

def path_h_cost(path):
  g_cost = 0
  for (node, cost) in path:
    g_cost += cost
    last_node = path[-1][0]
    h_cost = H_table[last_node]
  f_cost = g_cost + h_cost
  return g_cost, f_cost

path = [('S', 0), ('A', 1), ('C', 5)]
print(path_h_cost(path))

path = [('S', 0), ('A', 1), ('B', 2)]
print(path_h_cost(path))

from types import new_class
def Greedy_best_search(graph, start, goal):
  visited = []
  queue = [[(start, 0)]]
  while queue:
    queue.sort(key=path_h_cost)
    path = queue.pop()
    node = path[-1][0]
    if node in visited:
      continue
    visited.append(node)
    if node == goal:
      return path
    else:
      adjacent_nodes = graph.get(node, [])
      for (node2, cost) in adjacent_nodes:
        new_path = path.copy()
        new_path.append((node2, cost))
        queue.append(new_path)

solution = Greedy_best_search(graph, 'S', 'G')
print('Solution is', solution)
print('Cost solution is', path_h_cost(solution)[0])

