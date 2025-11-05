# Complex Networks Exam Notes

## Table of Contents
1. [NetworkX (Network Analysis)](#networkx-network-analysis)
2. [Matplotlib (Visualization)](#matplotlib-visualization)
3. [NumPy (Numerical Computing)](#numpy-numerical-computing)
4. [SciPy (Scientific Computing)](#scipy-scientific-computing)
5. [Network Models](#network-models)
6. [Network Metrics](#network-metrics)
7. [Community Detection](#community-detection)
8. [ODE Models](#ode-models)
9. [Boolean Networks](#boolean-networks)
10. [Agent-Based Simulations](#agent-based-simulations)
11. [Sampling Methods](#sampling-methods)

---

## NetworkX (Network Analysis)

### Basic Graph Creation

```python
import networkx as nx

# Create empty graph
G = nx.Graph()  # Undirected
G = nx.DiGraph()  # Directed

# Add edges
G.add_edge("A", "B")
G.add_edge("B", "C")
G.add_edges_from([(1, 2), (2, 3)])

# Add nodes
G.add_node("X")
G.add_nodes_from([1, 2, 3])

# From adjacency matrix
A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
G = nx.from_numpy_array(A)
G = nx.from_numpy_array(A, create_using=nx.DiGraph)  # Directed

# Relabel nodes
G = nx.relabel_nodes(G, lambda x: x+1)  # Add 1 to all node labels
```

### Reading/Writing Graphs

```python
# Read edge list
G = nx.read_edgelist('file.edges', comments='%', nodetype=int)

# Read DOT file
G = nx.Graph(nx.nx_pydot.read_dot('my.dot'))
G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='sorted', label_attribute='original_label')

# Write edge list
nx.write_edgelist(G, 'file.csv', data=False)

# Write GEXF
nx.write_gexf(G, "file.gexf", encoding='utf-8', prettyprint=True)
```

### Graph Properties

```python
# Basic properties
G.number_of_nodes()
G.number_of_edges()
G.nodes()  # Returns NodeView
G.edges()  # Returns EdgeView
G.degree()  # Returns DegreeView

# Convert to lists/dicts
list(G.nodes())
list(G.edges())
dict(G.degree())  # {node: degree}
list(dict(G.degree()).values())  # List of degrees

# Neighbors
G.neighbors(node)  # Iterator
list(G.neighbors(node))  # List of neighbors

# Adjacency matrix
A = nx.adjacency_matrix(G).todense()  # Sparse to dense

# Check edges
G.has_edge(u, v)
```

### Network Models

```python
# Erdős-Rényi
G = nx.erdos_renyi_graph(n=50, p=0.03, seed=19)
G = nx.erdos_renyi_graph(n=1000, p=2/999)  # p = <k>/(n-1)

# Barabási-Albert
G = nx.barabasi_albert_graph(n=100, m=1)  # m = edges per new node
G = nx.barabasi_albert_graph(n=10000, m=2)

# Stochastic Block Model
sizes = [20, 20]  # Community sizes
probs = [[0.5, 0.02], [0.02, 0.5]]  # Within/between probabilities
G = nx.stochastic_block_model(sizes, probs)

# For 3 communities
sizes = [10, 10, 5]
probs = [[0.8, 0.2, 0.1], [0.2, 0.8, 0.3], [0.1, 0.3, 0.8]]
G = nx.stochastic_block_model(sizes, probs, seed=0)
```

### Graph Operations

```python
# Connected components
components = list(nx.connected_components(G))
largest_component = max(components, key=len)
G_largest = G.subgraph(largest_component)

# Subgraph
G_sub = G.subgraph([1, 2, 3])  # Induced subgraph
G_sub = G.edge_subgraph([(1, 2), (2, 3)])  # Edge subgraph

# Copy
G_copy = G.copy()

# Remove nodes/edges
G.remove_node(node)
G.remove_nodes_from([1, 2, 3])
G.remove_edge(u, v)
G.clear_edges()
```

### Network Metrics

```python
# Degree distribution
degrees = list(dict(G.degree()).values())

# Clustering
clustering = nx.clustering(G)  # Dict: {node: coefficient}
avg_clustering = nx.average_clustering(G)  # Scalar

# Path lengths
diameter = nx.diameter(G)  # Requires connected graph
diameter = nx.diameter(G_largest)  # On largest component

# Centrality measures
degree_cent = nx.degree_centrality(G)  # Dict: {node: centrality}
betweenness_cent = nx.betweenness_centrality(G)
closeness_cent = nx.closeness_centrality(G)
harmonic_cent = nx.harmonic_centrality(G)

# Assortativity (average neighbor degree)
maxdegree = max(list(dict(G.degree()).values()))
knn = np.zeros(maxdegree + 1)
num = np.zeros(maxdegree + 1, dtype=int)
for i in G.nodes():
    neighdeg = [G.degree(j) for j in G.neighbors(i)]
    k = G.degree(i)
    num[k] += 1
    knn[k] += np.array(neighdeg).mean()
knn_normalized = knn / num  # Watch for division by zero!
```

---

## Matplotlib (Visualization)

### Basic Plotting

```python
import matplotlib.pyplot as plt
import numpy as np

# Basic plot
plt.plot(x, y)
plt.xlabel('Label')
plt.ylabel('Label')
plt.title('Title')
plt.legend()
plt.show()

# Multiple plots
plt.plot(x, y1, label='Series 1')
plt.plot(x, y2, label='Series 2')
plt.legend()

# Subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(x, y1)
ax2.plot(x, y2)
plt.tight_layout()
plt.show()

# Alternative subplot syntax
plt.subplot(1, 2, 1)
plt.plot(x, y1)
plt.subplot(1, 2, 2)
plt.plot(x, y2)
plt.tight_layout()
```

### Network Visualization

```python
# Basic drawing
nx.draw(G)
nx.draw(G, with_labels=True)
nx.draw(G, with_labels=True, node_color='yellow')

# With positions
pos = nx.spring_layout(G)
pos = nx.spring_layout(G, scale=10, gravity=0.01)  # Adjust spacing
nx.draw(G, pos=pos, with_labels=True)

# Colored nodes
node_colors = [0, 1, 1, 2, 2]  # List matching node order
nx.draw(G, node_color=node_colors, cmap=plt.cm.YlGn)

# Node sizes by degree
degrees = dict(G.degree())
nx.draw(G, nodelist=degrees.keys(), node_size=[v * 100 for v in degrees.values()])

# Graphviz layout (requires graphviz)
try:
    from networkx.drawing.nx_agraph import graphviz_layout
    pos = graphviz_layout(G, prog="dot")
except:
    from networkx.drawing.nx_pydot import graphviz_layout
    pos = graphviz_layout(G, prog="dot")

# Directed graphs
nx.draw(G, pos=pos, arrows=True, arrowstyle='-|>', arrowsize=12)

# Custom edge/node drawing
nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
nx.draw_networkx_nodes(G, pos, node_color='blue', node_size=500)
nx.draw_networkx_labels(G, pos, font_size=8)
plt.axis('off')
plt.show()
```

### Histograms and Distributions

```python
# Histogram
plt.hist(data, bins=10)
plt.hist(data, bins=bins, density=True, rwidth=0.8)

# Custom bins
bins = np.arange(max_value + 1) + 0.5
pos = np.arange(max_value) + 1
h, bins_out = np.histogram(data, bins=bins, density=True)
plt.bar(pos, h[0])

# Log-log plots
plt.loglog(x, y, 'r*')
plt.loglog(x, y, 'r*', label='Data')
plt.loglog(x, fitted, 'g-', label='Fit')

# Cumulative distribution
hcum = np.zeros_like(h[0])
for i in range(len(hcum)):
    hcum[i] = h[0][i:].sum()
plt.loglog(pos, hcum, 'r*')

# Logarithmic binning
log_bins = np.logspace(np.log10(0.9), np.log10(max_degree + 1), 15)
log_pos = (log_bins[1:]*2 + log_bins[:-1])/3
h_log = np.histogram(data, bins=log_bins, density=True)
```

### Animations

```python
from matplotlib.animation import FuncAnimation

# Simple animation
fig, ax = plt.subplots()
scat = ax.scatter(positions[:, 0], positions[:, 1])
ax.set_xlim(0, L)
ax.set_ylim(0, L)

def animate(frame):
    update_positions()
    scat.set_offsets(positions)
    return scat,

ani = FuncAnimation(fig, animate, frames=num_steps, interval=50, blit=True)

# For Jupyter
from IPython.display import HTML
HTML(ani.to_jshtml())

# Image animation
fig = plt.figure()
ims = []
for s in replay:
    im = plt.imshow(s, animated=True, vmin=0, vmax=2, cmap='rainbow')
    plt.axis('off')
    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
```

### Styling

```python
# Seaborn style
plt.style.use('seaborn-v0_8-poster')

# Figure size
plt.figure(figsize=(12, 6))

# Grid
plt.grid(True, alpha=0.3)

# Fill between (for error bars)
plt.fill_between(x, y - std, y + std, alpha=0.2)

# 3D plots
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
```

---

## NumPy (Numerical Computing)

### Arrays and Operations

```python
import numpy as np

# Array creation
arr = np.zeros((N, 4), dtype=int)
arr = np.ones(N)
arr = np.random.rand(N, 2) * L
arr = np.arange(0, 1.01, 0.01)
arr = np.linspace(0, 1, 100)

# Array operations
arr.shape
arr.reshape(L, L)
arr.copy()

# Indexing
arr[0]  # First element
arr[-1]  # Last element
arr[:, 0]  # First column
arr[0, :]  # First row

# Boolean indexing
mask = arr > threshold
arr[mask]
np.where(arr == value)[0]  # Returns indices

# Mathematical operations
np.sum(arr)
np.mean(arr)
np.std(arr)
np.max(arr)
np.min(arr)
np.argmax(arr)
np.argmin(arr)

# Element-wise operations
arr1 * arr2  # Element-wise multiplication
arr**2  # Element-wise power
np.sqrt(arr)
np.log(arr)
np.exp(arr)
```

### Random Numbers

```python
# Random arrays
np.random.rand(N)  # Uniform [0, 1)
np.random.rand(N, 2)  # Nx2 array
np.random.random(size=shape)  # Same as rand

# Random integers
np.random.randint(0, N)  # Single value
np.random.randint(0, N, size=k)  # Array

# Random choice
np.random.choice(range(N), 2, replace=False)  # Without replacement
np.random.choice(N, k, replace=False)

# Seeding
np.random.seed(42)
```

### Array Manipulation

```python
# Concatenation
arr = np.concatenate([arr1, arr2])

# Stacking
arr = np.vstack([arr1, arr2])  # Vertical
arr = np.hstack([arr1, arr2])  # Horizontal

# Reshaping
arr.reshape((L, L))
arr.flatten()

# Transpose
arr.T

# Matrix operations
np.dot(A, B)  # Matrix multiplication
A.dot(B)  # Alternative syntax
np.outer(a, b)  # Outer product
np.linalg.norm(arr)  # Vector norm
```

### Working with Sparse Data

```python
# Create array for neighbors (with padding)
max_degree = max(dict(G.degree()).values())
net = np.zeros((N, max_degree), dtype=int)

# Fill neighbor array
for i in range(N):
    neighbors = list(G.neighbors(i))
    for j, nei in enumerate(neighbors):
        net[i][j] = nei
    # Pad with 0s if needed (node 0 not in network)
```

---

## SciPy (Scientific Computing)

### ODE Solving

```python
from scipy.integrate import solve_ivp

# Define ODE system
def F(t, s):
    x, y = s[0], s[1]
    return [
        alpha * x - beta * x * y,
        delta * x * y - gamma * y
    ]

# Solve
t_eval = np.arange(0, 50.01, 0.01)
sol = solve_ivp(F, t_span=[0, 50], y0=[100, 3], t_eval=t_eval, 
                 atol=1e-8, rtol=1e-8)

# Access solution
sol.t  # Time points
sol.y  # State array (shape: [n_vars, n_timepoints])
sol.y[0]  # First variable over time
sol.y[1]  # Second variable over time

# Plot
plt.plot(sol.t, sol.y[0], label='x')
plt.plot(sol.t, sol.y[1], label='y')
```

### Curve Fitting

```python
from scipy.optimize import curve_fit

# Define function
def lin(x, a, b):
    return a * x + b

# Fit
# For power law: fit log(y) = a*log(x) + b
cond = h[0] > 0
fitdata = curve_fit(lin, np.log(pos[cond]), np.log(h[0][cond]))
params, covariance = fitdata
a, b = params[0], params[1]

# Evaluate fit
y_fit = pos**a * np.exp(b)  # For log-log fit
# Or: y_fit = lin(x, a, b)

# Plot
plt.loglog(pos, h[0], 'r*')
plt.loglog(pos, y_fit, 'g-')
```

### Statistics

```python
from scipy.stats import poisson, pearsonr

# Poisson distribution
mean = np.mean(degrees)
x = np.arange(9)
poisson_pmf = poisson.pmf(x, mu=mean)

# Correlation
corr = np.corrcoef(h, poisson_pmf)[0, 1]
corr = pearsonr(h, poisson_pmf)[0]
```

---

## Network Models

### Erdős-Rényi (ER)

```python
# Create ER graph
N = 1000
p = 0.01  # Probability of edge
G = nx.erdos_renyi_graph(N, p)

# Expected degree: <k> = p * (N-1)
# For fixed <k>: p = <k> / (N-1)
p = 2 / (N - 1)
G = nx.erdos_renyi_graph(N, p)

# Degree distribution: Poisson
degrees = list(dict(G.degree()).values())
mean_degree = np.mean(degrees)
```

### Barabási-Albert (BA)

```python
# Create BA graph
n = 1000
m = 2  # Edges per new node
G = nx.barabasi_albert_graph(n, m)

# Degree distribution: Power law P(k) ~ k^(-3)
# Cumulative: P(K>k) ~ k^(-2)
```

### Stochastic Block Model (SBM)

```python
# Two communities
sizes = [20, 20]
P_same = 0.5
P_other = 0.02
probs = [[P_same, P_other], [P_other, P_same]]
G = nx.stochastic_block_model(sizes, probs)

# Three communities
sizes = [10, 10, 5]
probs = [[0.8, 0.2, 0.1], [0.2, 0.8, 0.3], [0.1, 0.3, 0.8]]
G = nx.stochastic_block_model(sizes, probs, seed=0)
```

---

## Network Metrics

### Degree Distribution

```python
# Get degrees
degrees = list(dict(G.degree()).values())

# Histogram
bins = np.arange(max(degrees) + 1) + 0.5
h, bins_out = np.histogram(degrees, bins=bins, density=True)
pos = np.arange(len(h)) + 1

# Plot
plt.bar(pos, h)
plt.xlabel('k')
plt.ylabel('p(k)')
```

### Power Law Fitting

```python
# Log-log plot
plt.loglog(pos, h[0], 'r*')

# Fit power law: P(k) ~ k^(-gamma)
# Fit log(P(k)) = -gamma * log(k) + constant
cond = h[0] > 0
fitdata = curve_fit(lin, np.log(pos[cond]), np.log(h[0][cond]))
gamma = -fitdata[0][0]  # Negative of slope

# Cumulative distribution
hcum = np.zeros_like(h[0])
for i in range(len(hcum)):
    hcum[i] = h[0][i:].sum()
# Cumulative exponent is typically gamma - 1
```

### Clustering Coefficient

```python
# Per-node clustering
clustering = nx.clustering(G)  # Dict

# Average clustering
avg_clustering = nx.average_clustering(G)

# Clustering by degree
clustering_by_degree = {}
for degree, nodes in nodes_with_degree.items():
    clustering_by_degree[degree] = np.mean([nx.clustering(G, node) for node in nodes])
```

### Assortativity

```python
# Average neighbor degree as function of degree
maxdegree = max(list(dict(G.degree()).values()))
knn = np.zeros(maxdegree + 1)
num = np.zeros(maxdegree + 1, dtype=int)

for i in G.nodes():
    neighdeg = [G.degree(j) for j in G.neighbors(i)]
    k = G.degree(i)
    num[k] += 1
    knn[k] += np.array(neighdeg).mean()

knn_normalized = knn / num  # Watch for division by zero!
plt.plot(knn_normalized, 'r*-')
plt.xlabel('k')
plt.ylabel('k_nn(k)')
```

---

## Community Detection

```python
from itertools import product

# Greedy Modularity
communities = nx.community.greedy_modularity_communities(G)
# Returns list of sets

# Girvan-Newman
generator = nx.community.girvan_newman(G)
communities = [c for c in next(generator)]
# Returns list of sets

# Louvain
communities = nx.community.louvain_communities(G)

# Label Propagation
communities = nx.community.label_propagation_communities(G)

# Helper: Convert to node->community dict
def get_node_community(G, communities):
    node_community = {}
    for i, community in enumerate(communities):
        for node in community:
            node_community[node] = i
    return node_community

# Visualize communities
node_community = get_node_community(G, communities)
nx.draw(G, pos=pos, node_color=[node_community[n] for n in G.nodes()])
```

### Comparing Partitions

```python
# Rand Index and Jaccard Index
def compute_rand_jaccard_indices(G, partitionA, partitionB):
    node_communityA = get_node_community(G, partitionA)
    node_communityB = get_node_community(G, partitionB)
    n = len(G.nodes)
    
    a_11 = 0  # Same in both
    a_10 = 0  # Same in A, different in B
    a_01 = 0  # Different in A, same in B
    a_00 = 0  # Different in both
    
    for u, v in product(range(n), range(n)):
        if u == v:
            continue
        if node_communityA[u] == node_communityA[v] and node_communityB[u] == node_communityB[v]:
            a_11 += 1
        elif node_communityA[u] == node_communityA[v] and node_communityB[u] != node_communityB[v]:
            a_10 += 1
        elif node_communityA[u] != node_communityA[v] and node_communityB[u] == node_communityB[v]:
            a_01 += 1
        else:
            a_00 += 1
    
    rand_index = (a_11 + a_00) / (a_11 + a_10 + a_01 + a_00)
    jaccard_index = a_11 / (a_11 + a_10 + a_01)
    return rand_index, jaccard_index
```

---

## ODE Models

### SIR Model

```python
def SIR(t, s):
    I, S = s[0], s[1]
    return [
        beta * I * S - mu * I,
        -beta * I * S
    ]

# R = N - I - S (total population)
```

### SIS Model

```python
def SIS(t, s):
    I, S = s[0], s[1]
    return [
        beta_t(t) * I * S - mu * I,
        -beta_t(t) * I * S + mu * I
    ]
```

### Lotka-Volterra (Predator-Prey)

```python
alpha = 2.0   # Prey growth
beta = 0.005  # Predation
delta = 0.005 # Predator growth from prey
gamma = 1.0   # Predator death

def F(t, s):
    x, y = s[0], s[1]  # Prey, Predator
    return [
        alpha * x - beta * x * y,
        delta * x * y - gamma * y
    ]
```

### Lorenz Attractor

```python
sigma = 10
rho = 28
beta = 8/3

def F2(t, s):
    x, y, z = s[0], s[1], s[2]
    return [
        sigma * (y - x),
        rho * x - x * z - y,
        x * y - beta * z
    ]

# 3D plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol.y[0], sol.y[1], sol.y[2])
```

---

## Boolean Networks

### Basic Boolean Network

```python
class BooleanNetwork:
    def __init__(self):
        self.state = {}
        # Initialize missing keys
        while True:
            try:
                self.update()
            except KeyError as e:
                missing_key = e.args[0]
                self.state[missing_key] = False
            else:
                break

    def update(self):
        new_state = self.state.copy()
        s = self.state
        
        # Update rules (synchronous)
        new_state["A"] = s["B"] and not s["C"]
        new_state["B"] = s["A"] or s["C"]
        # ...
        
        self.state = new_state

    def run(self, steps):
        for _ in range(steps):
            self.update()
```

### Asynchronous Updates

```python
import random

def async_step_random(net, clamp, candidate_nodes):
    """Pick random node and update only that node."""
    node = random.choice(candidate_nodes)
    
    old_state = net.state.copy()
    net.update()
    new_state = net.state.copy()
    
    # Apply only selected node's update
    for k in new_state.keys():
        if k != node:
            net.state[k] = old_state.get(k, False)
    
    # Enforce clamped inputs
    for k, v in clamp.items():
        net.state[k] = bool(v)
```

### Finding Attractors

```python
def state_fingerprint(state):
    """Create hashable fingerprint."""
    node_order = sorted(state.keys())
    return tuple(bool(state.get(k, False)) for k in node_order)

def run_to_attractor(clamp, max_steps=2000):
    net = BooleanNetwork()
    for k, v in clamp.items():
        net.state[k] = bool(v)
    
    seen = {}
    history = []
    
    fp = state_fingerprint(net.state)
    seen[fp] = 0
    history.append(net.state.copy())
    
    for step in range(1, max_steps + 1):
        net.update()
        for k, v in clamp.items():
            net.state[k] = bool(v)
        
        fp = state_fingerprint(net.state)
        if fp in seen:
            start = seen[fp]
            cycle_len = step - start
            return start, cycle_len, history + [net.state.copy()]
        seen[fp] = step
        history.append(net.state.copy())
    
    return len(history) - 1, 0, history
```

### Extracting Network Structure

```python
import ast
import inspect
import textwrap

class DictKeyExtractor(ast.NodeVisitor):
    def __init__(self):
        self.target_key = None
        self.source_keys = set()

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Subscript):
            if isinstance(node.targets[0].slice, ast.Constant):
                self.target_key = node.targets[0].slice.value
        self.visit(node.value)

    def visit_Subscript(self, node):
        if isinstance(node.slice, ast.Constant):
            self.source_keys.add(node.slice.value)

def extract_keys_line_by_line(func):
    source = inspect.getsource(func)
    source = textwrap.dedent(source)
    lines = source.split("\n")
    
    results = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            tree = ast.parse(line)
            extractor = DictKeyExtractor()
            extractor.visit(tree)
            if extractor.target_key:
                results.append((line, extractor.target_key, extractor.source_keys))
        except SyntaxError:
            continue
    return results

# Build NetworkX graph from rules
keys_by_line = extract_keys_line_by_line(BooleanNetwork.update)
G = nx.DiGraph()
for _, target_key, source_keys in keys_by_line:
    G.add_node(target_key)
    for src in source_keys:
        G.add_node(src)
        G.add_edge(src, target_key)
```

---

## Agent-Based Simulations

### Basic Agent Setup

```python
# Initialize agents
num_agents = 100
L = 10  # Size of area
positions = np.random.rand(num_agents, 2) * L
directions = np.random.rand(num_agents) * 2 * np.pi

# Update positions
new_positions = positions + v * np.array([np.cos(directions), np.sin(directions)]).T
new_positions %= L  # Periodic boundary
```

### K-Nearest Neighbors

```python
import heapq

def get_k_closest_neighbors(agent_id, k, positions, L):
    heap = []
    for i in range(len(positions)):
        if i != agent_id:
            # Periodic boundary distance
            dx = positions[agent_id][0] - positions[i][0]
            dy = positions[agent_id][1] - positions[i][1]
            dx = dx - L * np.round(dx / L)
            dy = dy - L * np.round(dy / L)
            distance = np.sqrt(dx**2 + dy**2)
            
            if len(heap) < k:
                heapq.heappush(heap, (-distance, i))
            elif distance < -heap[0][0]:
                heapq.heapreplace(heap, (-distance, i))
    
    return [n for _, n in heap]
```

### Alignment/Vicsek Model

```python
def update_directions(positions, directions, neighbors, align_strength, noise_strength):
    new_directions = np.zeros_like(directions)
    
    for i in range(len(positions)):
        if neighbors[i]:
            neighbor_vectors = np.array([[np.cos(directions[j]), np.sin(directions[j])] 
                                        for j in neighbors[i]])
            sum_vector = np.sum(neighbor_vectors, axis=0)
            align_angle = np.arctan2(sum_vector[1], sum_vector[0])
            
            # Weighted combination
            orig_angle = directions[i]
            noise_angle = np.random.rand() * 2 * np.pi
            
            new_directions[i] = ((1 - align_strength - noise_strength) * orig_angle +
                                align_strength * align_angle +
                                noise_strength * noise_angle)
        else:
            new_directions[i] = directions[i]
    
    return new_directions
```

### SIRS on Lattice

```python
# States: S=0, I=1, R=2
L = 100
N = L * L
T = np.zeros(N, dtype=int)
T[0] = I  # Initial infected

# Neighbor array for square lattice
net = np.zeros((N, 4), dtype=int)
for i in range(N):
    x = i % L
    y = i // L
    net[i][0] = (x+1) % L + y*L  # Right
    net[i][1] = (x-1) % L + y*L  # Left
    net[i][2] = x % L + ((y+1) % L) * L  # Up
    net[i][3] = x % L + ((y-1) % L) * L  # Down

# Update step
infected = np.where(T == I)[0]
infected_neighs = net[infected]

# Infection: S neighbors of I become I with probability p_t
mask = (T[infected_neighs] == S) * (p_t > np.random.random(size=infected_neighs.shape))
T[infected_neighs[mask]] = I

# Recovery: I becomes R with probability p_r
recover_mask = np.random.random(size=infected.shape) < p_r
T[infected[recover_mask]] = R
```

---

## Sampling Methods

### Random Node Sampling

```python
# Sample M nodes from N
M = 20
N = 60
node_s = np.random.choice(N, M, replace=False)
node_s.sort()

# Use sampled nodes
G_sub = G.subgraph(node_s)
pos2 = {k: v for k, v in pos.items() if k in node_s}
nx.draw(G_sub, pos=pos2, node_color=c[node_s])
```

### Random Walk Sampling

```python
M = 20
i = np.random.randint(N)
node_s = set([i])

while len(node_s) < M:
    neighs = list(G.neighbors(i))
    i = np.random.choice(neighs, 1)[0]
    node_s.add(i)

node_s = np.array(list(node_s))
node_s.sort()
```

### Edge Sampling

```python
# Sample edges until M nodes
edges = list(G.edges())
node_s = set()
sampled_edges = []

while len(node_s) < M:
    i = np.random.choice(len(edges), 1)[0]
    edge = edges.pop(i)
    sampled_edges.append(edge)
    node_s.add(edge[0])
    node_s.add(edge[1])

# Use edge subgraph
G_sub = G.edge_subgraph(sampled_edges)
# Or induced subgraph
G_sub = G.subgraph(node_s)
```

### Affinity-Based Edge Sampling

```python
# Assign affinities
affinities = np.random.exponential(1.0, N)

# Normalize
aff_min = affinities.min()
aff_max = affinities.max()
aff_norm = (affinities - aff_min) / (aff_max - aff_min + 1e-12)

# Edge probability proportional to min affinity
edges = list(G.edges())
p_edge = np.array([min(aff_norm[u], aff_norm[v]) for u, v in edges])

# Scale to target fraction
target_fraction = 0.3
scale = target_fraction / p_edge.mean()
p_edge = np.clip(scale * p_edge, 0.0, 1.0)

# Sample edges
keep_mask = np.random.rand(len(edges)) < p_edge
kept_edges = [e for e, keep in zip(edges, keep_mask) if keep]

G_sampled = nx.Graph()
G_sampled.add_nodes_from(G.nodes())
G_sampled.add_edges_from(kept_edges)
```

---

## Centrality and Node Removal

### Node Removal by Centrality

```python
def P_infty(G):
    """Probability node belongs to giant component."""
    if G.number_of_nodes() == 0:
        return 0
    components = list(nx.connected_components(G))
    largest = max(components, key=len)
    return len(largest) / G.number_of_nodes()

def S(G):
    """Average size of non-giant components."""
    if G.number_of_nodes() == 0:
        return 0
    components = list(nx.connected_components(G))
    largest = max(components, key=len)
    components.remove(largest)
    if len(components) == 0:
        return 0
    return np.mean([len(c) for c in components])

# Remove nodes by centrality
def remove_by_centrality(G, centrality_func, ascending=True):
    centrality = centrality_func(G)
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=not ascending)
    
    G_copy = G.copy()
    N = G_copy.number_of_nodes()
    P_infty_values = []
    S_values = []
    fractions = []
    
    for node, _ in sorted_nodes:
        P_infty_values.append(P_infty(G_copy))
        S_values.append(S(G_copy))
        fractions.append((N - G_copy.number_of_nodes()) / N)
        G_copy.remove_node(node)
    
    return fractions, P_infty_values, S_values

# Plot
fractions, P_infty_vals, S_vals = remove_by_centrality(G, nx.degree_centrality, ascending=False)
plt.plot(fractions, P_infty_vals, label='P_infty')
plt.plot(fractions, S_vals, label='S')
```

---

## Common Patterns

### Ensemble Averaging

```python
nsamples = 10000
results = []

for _ in range(nsamples):
    # Generate network/run simulation
    G = nx.erdos_renyi_graph(N, p)
    # Compute metric
    metric = compute_metric(G)
    results.append(metric)

# Average
mean_result = np.mean(results, axis=0)
std_result = np.std(results, axis=0)
```

### Coefficient of Determination (R²)

```python
# SS_res: Sum of squares of residuals
ss_res = np.sum((y_observed - y_fitted)**2)

# SS_tot: Total sum of squares
ss_tot = np.sum((y_observed - np.mean(y_observed))**2)

# R²
R_squared = 1 - (ss_res / ss_tot)

# Or use correlation
R_squared = np.corrcoef(y_observed, y_fitted)[0, 1]**2
```

### Working with Collections

```python
from collections import Counter

# Count occurrences
word_counts = Counter(words)
degree_counts = Counter(degrees)

# Convert to sorted list
sorted_counts = sorted(word_counts.values(), reverse=True)

# Convert to array
counts_array = np.array(sorted_counts)
```

---

## Quick Reference

### Common Imports
```python
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.stats import poisson
from collections import Counter
from itertools import product
import heapq
```

### NetworkX Quick Reference
- `G = nx.Graph()` / `nx.DiGraph()` - Create graph
- `G.add_edge(u, v)` - Add edge
- `G.degree()` - Get degrees
- `nx.erdos_renyi_graph(n, p)` - ER model
- `nx.barabasi_albert_graph(n, m)` - BA model
- `nx.stochastic_block_model(sizes, probs)` - SBM
- `nx.clustering(G)` - Clustering coefficients
- `nx.connected_components(G)` - Components
- `nx.spring_layout(G)` - Layout
- `nx.draw(G, pos=pos)` - Draw graph

### Matplotlib Quick Reference
- `plt.plot(x, y)` - Line plot
- `plt.scatter(x, y)` - Scatter plot
- `plt.hist(data, bins=10)` - Histogram
- `plt.loglog(x, y)` - Log-log plot
- `plt.subplot(1, 2, 1)` - Subplot
- `plt.xlabel('Label')` - Axis label
- `plt.legend()` - Show legend
- `plt.show()` - Display plot

### NumPy Quick Reference
- `np.zeros(shape)` - Zeros array
- `np.random.rand(N)` - Random array
- `np.arange(start, stop, step)` - Range array
- `np.where(condition)[0]` - Find indices
- `arr.reshape(L, L)` - Reshape
- `np.mean(arr)` - Mean
- `np.std(arr)` - Standard deviation

---

## Tips for Exams

1. **NetworkX**: Remember to convert views to lists/dicts when needed (`list(G.nodes())`, `dict(G.degree())`)

2. **Plotting**: Always check if you need `plt.show()` or `plt.tight_layout()`

3. **Array Operations**: Use boolean indexing for filtering (`arr[mask]`)

4. **ODE Solving**: `solve_ivp` returns `sol.y` as shape `[n_vars, n_timepoints]`, access with `sol.y[0]`

5. **Power Law Fitting**: Fit on log-log scale, remember to exclude zeros when fitting

6. **Subgraphs**: Use `G.subgraph(nodes)` for induced subgraph, `G.edge_subgraph(edges)` for edge subgraph

7. **Periodic Boundaries**: Use modulo for wrapping (`arr % L`)

8. **Heap for K-Nearest**: Use negative distances for max-heap behavior (`heapq.heappush(heap, (-dist, i))`)

