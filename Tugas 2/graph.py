import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('C:/Users/zeroo/OneDrive/Documents/Project Mandiri/AI (Python)/Tugas 2/graph.png')
plt.imshow(img)
plt.show()

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

# visited = []
# queue = []

visited = []

def dfs(visited, graph, node):
  if node not in visited:
    print(node)
    visited.append(node)
    for neighbour in graph[node]:
      dfs(visited, graph, neighbour)

# def bfs(visited, graph, node):
#     visited.append(node)
#     queue.append(node)
#     while queue:
#         s = queue.pop(0)
#         print(s, end=' ')
#         for neighbour in graph[s]:
#             if neighbour not in visited:
#                 visited.append(neighbour)
#                 queue.append(neighbour)

# bfs(visited, graph, 'A')

dfs(visited, graph, 'A')
