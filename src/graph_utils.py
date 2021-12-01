import networkx as nx
import random
import numpy as np
import itertools
import matplotlib.pyplot as plt

import scipy as sp
import scipy.sparse  # call as sp.sparse
import scipy.sparse.linalg  # call as sp.sparse.linalg


"""
DRAWING FUNCTIONS
"""

'''def _dense_spectral(G:nx.Graph, weight:str):
    A = nx.to_scipy_sparse_matrix(G, weight=weight, dtype="d")
    nnodes, _ = A.shape

    D = np.identity(nnodes, dtype=A.dtype) * np.sum(A, axis=1)
    L = D - A
    eigenvalues, eigenvectors = np.linalg.eig(L)

    # sort and keep smallest nonzero
    index = np.argsort(eigenvalues)[1 : 2 + 1]  # 0 index is zero eigenvalue
    return np.real(eigenvectors[:, index])


def _sparse_spectral(G:nx.Graph, weight:str, seed:int=None):
    A = nx.to_scipy_sparse_matrix(G, weight=weight, dtype="d")
    nnodes, _ = A.shape

    data = np.asarray(A.sum(axis=1).T)
    D = sp.sparse.spdiags(data, 0, nnodes, nnodes)
    L = D - A

    k = 2 + 1
    # number of Lanczos vectors for ARPACK solver.What is the right scaling?
    ncv = max(2 * k + 1, int(np.sqrt(nnodes)))
    # return smallest k eigenvalues and eigenvectors
    if seed is not None:
        np.random.seed(seed)
    v0 = np.random.uniform(0, 1, (2, nnodes))
    eigenvalues, eigenvectors = sp.sparse.linalg.eigen.eigsh(L, k, which="SM", ncv=ncv, v0=v0)
    index = np.argsort(eigenvalues)[1:k]  # 0 index is zero eigenvalue
    return np.real(eigenvectors[:, index])'''


'''def _spectral_layout(G: nx.Graph, weight:str=None, seed:int=None):
    
    nnodes = len(G.nodes)

    pos = []
    if nnodes < 500:
        pos = _dense_spectral(G, weight)
    else:
        pos = _sparse_spectral(G, weight, seed)

    pos = nx.rescale_layout(pos, scale=1) + np.zeros(2)
    pos = dict(zip(G, pos))
    return pos'''


def prettyPos(G: nx.Graph, weight:str=None, weight_type:str='both', seed:int=None) -> dict:
    """ Obtain the position for spectral -> spring 
    
    Attributes
    ----------
    G : nx.Graph
        Graph to draw.

    weight: str
        Name of the edge attribute to use as weight.
    
    weight_type : str
        One in ['both', 'spectral', 'spring'].
    """
    pos = None
    np.random.seed(seed)
    
    if weight_type == 'spring':
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spectral_layout(G, weight=weight)

    if weight_type == 'spectral':
        pos = nx.spring_layout(G, pos=pos)
    else:
        pos = nx.spring_layout(G, pos=pos, weight=weight)
    return pos

def prettyDraw(G: nx.Graph, measures=False, **kwargs) -> None:
    """ Draw graph using spring layout and initial position of spectral layout"""
    pos = prettyPos(G)
    nx.draw(G, pos=pos, **kwargs)

    if measures:
        quality_measures(G, pos)
    

def drawRelaxingEdges(G: nx.Graph, edges: list, measures=False, **kwargs) -> None:
    """ Draw the graph relaxing some edges """
    tempG = G.copy()
    tempG.remove_edges_from(edges)
    pos = prettyPos(tempG)
    tempG.add_edges_from(edges)

    colors = ['r' if e in edges else 'b' for e in G.edges]

    nx.draw(tempG, pos=pos, edge_color=colors, **kwargs)

    if measures:
        quality_measures(G, pos)



"""
DRAWING QUALITY MEASURES
"""
def _same_side(pos, p, q, a, b):
    """Indicates whether the points a and b are at the same side of segment p-q"""
    dx = pos[p][0] - pos[q][0]
    dy = pos[p][1] - pos[q][1]
    dxa = pos[a][0] - pos[p][0]
    dya = pos[a][1] - pos[p][1]
    dxb = pos[b][0] - pos[p][0]
    dyb = pos[b][1] - pos[p][1]
    return (dy*dxa - dx*dya > 0) == (dy*dxb - dx*dyb > 0)


def _edge_crossing(pos, e1, e2):
    """Returns True if edges e1 and e2 cross, and False otherwise."""
    a, b, c, d = e1[0], e1[1], e2[0], e2[1]
    ax, ay, bx, by = pos[a][0], pos[a][1], pos[b][0], pos[b][1]
    cx, cy, dx, dy = pos[c][0], pos[c][1], pos[d][0], pos[d][1]
    if min(ax, bx) > max(cx, dx) or max(ax, bx) < min(cx, dx):
        return False
    if min(ay, by) > max(cy, dy) or max(ay, by) < min(cy, dy):
        return False
    if a == c or a == d or b == c or b == d:
        return False
    return not (_same_side(pos, a, b, c, d) or _same_side(pos, c, d, a, b))


def _angle_segments(s1x:tuple, s1y:tuple, s2x:tuple, s2y:tuple) -> float:
    """Returns the angle between two segments"""
    x1, x2 = np.array(s1y) - np.array(s1x)
    y1, y2 = np.array(s2y) - np.array(s2x)
    cos = (x1 * y1 + x2 * y2) / ( np.sqrt(x1*x1 + x2*x2) * np.sqrt(y1*y1 + y2*y2) )
    angle = np.arccos(cos)*180/np.pi
    return min(180-angle, angle)

def _angle_segments2(s1x:tuple, s1y:tuple, s2x:tuple, s2y:tuple) -> float:
    """Returns the angle between two segments"""
    x1, x2 = np.array(s1y) - np.array(s1x)
    y1, y2 = np.array(s2y) - np.array(s2x)
    cos = (x1 * y1 + x2 * y2) / ( np.sqrt(x1*x1 + x2*x2) * np.sqrt(y1*y1 + y2*y2) )
    angle = np.arccos(cos)*180/np.pi
    return angle


def _euclidean_dist(x:tuple, y:tuple) -> float:
    """Euclidean distance"""
    return np.sqrt((y[0] - x[0])**2 + (y[1] - x[1])**2)



def stress(G, pos):
    val = 0
    for n1 in G.nodes:
        for n2 in G.nodes:
            if n2 >= n1:
                break
            ideal = len(nx.algorithms.shortest_paths.generic.shortest_path(G, n1, n2))
            p1, p2 = np.array(pos[n1]), np.array(pos[n2])
            val += (ideal**(-2))*(_euclidean_dist(p1,p2) - ideal)**2

    return val


def edge_length_variance(G, pos):
    """Returns the variance of the edge lenghts"""
    pos = nx.drawing.layout.rescale_layout_dict(pos)
    edge_lengths = [_euclidean_dist(pos[u], pos[v]) for u,v in G.edges]
    return np.var(edge_lengths)


def mean_edge_length(G, pos):
    """Returns the variance of the edge lenghts"""
    pos = nx.drawing.layout.rescale_layout_dict(pos)
    edge_lengths = [_euclidean_dist(pos[u], pos[v]) for u,v in G.edges]
    return np.mean(edge_lengths)


def num_crossings(G, pos):
    """Returns the number of edge crossings."""
    # Clean the attribute
    for u, v in list(G.edges):
        G[u][v]['num_cross'] = 0

    n = 0
    for e1, e2 in list(itertools.combinations(G.edges, 2)):
        if _edge_crossing(pos, e1, e2):
            n += 1
            G[e1[0]][e1[1]]['num_cross'] += 1
            G[e2[0]][e2[1]]['num_cross'] += 1
    return n



def mean_crossing_angle(G, pos):
    """Returns the mean angle of the crossings."""
    n = 0
    angle = 0
    for e1, e2 in list(itertools.combinations(G.edges, 2)):
        if _edge_crossing(pos, e1, e2):
            e1x, e1y = pos[e1[0]], pos[e1[1]]
            e2x, e2y = pos[e2[0]], pos[e2[1]]
            n += 1
            angle += _angle_segments(e1x, e1y, e2x, e2y)

            
    return angle/n if n != 0 else 0


def aspect_ratio(pos):
    """Returns the aspect ratio."""
    coords = np.array(list(pos.values()))

    maxx, maxy = np.max(coords, axis=0)
    minx, miny = np.min(coords, axis=0)

    w = maxx - minx
    h = maxy - miny
    return min(w, h)/max(w, h)


def pseudo_vertex_resolution(G, pos):
    """Min distance to other node averaged out for all nodes
    
    Used to determine the amount of clutter and overlapping nodes.
    Note: the positions are normalized before computing the measure
    """
    pos = nx.drawing.layout.rescale_layout_dict(pos)

    total_min_dist = 0
    for n1 in G.nodes:
        min_dist = np.inf
        for n2 in G[n1]:
            min_dist = min(min_dist, _euclidean_dist(pos[n1], pos[n2]))
        total_min_dist += min_dist
    return total_min_dist/len(G.nodes)


def mean_angular_resolution(G, pos):
    """Min angle between outgoing edges averaged out for all nodes

    Used to determine the clarity of the edges
    """
    total_angles = 0
    for n1 in G.nodes:
        min_angle = np.inf
        for n2 in G[n1]:
            for n3 in G[n1]:
                if n2 < n3:
                    p11, p12 = pos[n1], pos[n2]
                    p21, p22 = pos[n1], pos[n3]
                    min_angle = min(min_angle, _angle_segments(p11,p12,p21,p22))
        total_angles += min_angle
    return total_angles/len(G.nodes)


def continuity2(G, pos, min_length=3, max_length=5):
    pos = nx.drawing.layout.rescale_layout_dict(pos)
    paths = dict(nx.all_pairs_shortest_path(G))
    lenghts = {(s,t): len(paths[s][t]) for s in paths.keys() for t in paths[s].keys()}
    node_pairs = [k for k,v in lenghts.items() if min_length <= v <= max_length]
    total_val = 0
    for u, v in node_pairs:
        opt_dist = _euclidean_dist(pos[u], pos[v])
        node_pairs_path = zip(paths[u][v][:-1], paths[u][v][1:])
        real_dist = np.sum([_euclidean_dist(pos[uu], pos[vv]) for uu, vv in node_pairs_path])
        total_val += real_dist/opt_dist
    return total_val


def continuity(G, pos, min_length=3, max_length=5):
    pos = nx.drawing.layout.rescale_layout_dict(pos)
    paths = dict(nx.all_pairs_shortest_path(G))
    lenghts = {(s,t): len(paths[s][t]) for s in paths.keys() for t in paths[s].keys()}
    node_pairs = [k for k,v in lenghts.items() if min_length <= v <= max_length]
    total_val = 0
    for u, v in node_pairs:
        node_trios_path = zip(paths[u][v][:-2], paths[u][v][1:-1], paths[u][v][2:])
        mean_angle = np.mean(
            [_angle_segments2(pos[uu], pos[vv], pos[vv], pos[ww]) for uu, vv, ww in node_trios_path]
            )
        total_val += mean_angle
    return total_val/len(node_pairs)


def bendiness_ratio(G, pos, min_length=3, max_length=5):
    pos = nx.drawing.layout.rescale_layout_dict(pos)
    paths = dict(nx.all_pairs_shortest_path(G))
    lenghts = {(s,t): len(paths[s][t]) for s in paths.keys() for t in paths[s].keys()}
    node_pairs = [k for k,v in lenghts.items() if min_length <= v <= max_length]
    total_val = 0
    for u, v in node_pairs:
        node_trios_path = zip(paths[u][v][:-2], paths[u][v][1:-1], paths[u][v][2:])
        node_pairs_path = zip(paths[u][v][:-1], paths[u][v][1:])
        sum_angle = np.sum(
            [_angle_segments2(pos[uu], pos[vv], pos[vv], pos[ww]) for uu, vv, ww in node_trios_path]
            )
        sum_dist = np.sum([_euclidean_dist(pos[uu], pos[vv]) for uu, vv in node_pairs_path])
        total_val += sum_angle/sum_dist
    return total_val/len(node_pairs)


def quality_measures(G, pos, show=False) -> list:
    """Returns a table with the quality measures"""
    if show:
        print(f"           Num crossings: {num_crossings(G, pos)}")
        print(f"            Aspect ratio: {aspect_ratio(pos)}")
        print(f"     Mean crossing angle: {mean_crossing_angle(G, pos)}")
        print(f"Pseudo vertex resolution: {pseudo_vertex_resolution(G, pos)}")
        print(f" Mean angular resolution: {mean_angular_resolution(G, pos)}")
    return num_crossings(G, pos), aspect_ratio(pos), mean_crossing_angle(G, pos), pseudo_vertex_resolution(G, pos), mean_angular_resolution(G, pos)



def compareGraphs(g1:nx.Graph, g2:nx.Graph, pos1:dict=None, pos2:dict=None, rmedges:list=[], show:bool=True) -> list:
    """Compares the metrics between 2 grafs and their drawings"""
    def symbol(val: float, desired_comp: str):
        if desired_comp == 'g':
            return '🟡' if (val == 0 or np.isnan(val)) else ('✅' if val > 0 else '❌')
        elif desired_comp == 'l':
            return '🟡' if (val == 0 or np.isnan(val)) else ('✅' if val < 0 else '❌')

    pos1 = pos1 if pos1 != None else prettyPos(g1)
    pos2 = pos2 if pos2 != None else prettyPos(g2)

    q1 = quality_measures(g1, pos1, show=False)
    q2 = quality_measures(g2, pos2, show=False)
    v = np.array(q2) - np.array(q1)
    if show:
        #print(f"           Edges removed: {len(rmedges)} ({len(rmedges)/len(g.edges)*100:.2f}%)")
        print(f"           Num crossings: {q1[0]} - {q2[0]} ({symbol(v[0], 'l')} {v[0]})")
        print(f"            Aspect ratio: {q1[1]:.3f} - {q2[1]:.3f} ({symbol(v[1], 'g')} {v[1]:.3f})")
        print(f"     Mean crossing angle: {q1[2]:.3f} - {q2[2]:.3f} ({symbol(v[2], 'g')} {v[2]:.3f})")
        print(f"Pseudo vertex resolution: {q1[3]:.3f} - {q2[3]:.3f} ({symbol(v[3], 'g')} {v[3]:.3f})")
        print(f" Mean angular resolution: {q1[4]:.3f} - {q2[4]:.3f} ({symbol(v[4], 'g')} {v[4]:.3f})")

        plt.subplot(121, aspect=1)
        nx.draw(g1, pos=pos1, node_size=20)
        plt.subplot(122, aspect=1)
        e_col = ['grey' if (e in rmedges or e[::-1] in rmedges) else 'black' for e in g2.edges]
        nx.draw(g2, pos=pos2, node_size=10, edge_color=e_col)

    return v


"""
EXTRA FUNCTIONS
"""
def addRandomEdges(graph: nx.Graph, nEdges: int) -> tuple:
    """ Adds random edges to a given graph """
    nodes = list(graph.nodes)
    n = len(nodes)
    edges = []
    for i in range(nEdges):
        newEdge = False
        while not newEdge:
            i_u, i_v = random.randint(0, n-1), random.randint(0, n-1)
            edge = [nodes[i_u], nodes[i_v]]
            if edge not in graph.edges:
                newEdge = True
        edges.append(edge)
    g = graph.copy()
    g.add_edges_from(edges)
    return g, edges




"""
KOREN
"""

def _normalize(v: np.array):
    norm = np.linalg.norm(v)
    norm = norm if norm != 0 else np.finfo(v.dtype).eps
    return v/norm

def _korenMatrix(G: nx.graph, weight = None) -> np.array:
    A = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
    if weight is not None:
        for (u,v),w in weight.items():
            A[u][v] = A[v][u] = w
    n = A.shape[0]
    D = np.diag(np.array([v for k,v in G.degree()]))
    D_inv = np.diag(np.array([1/v for k,v in G.degree()]))
    I = np.identity(n)

    return 0.5 * (I + np.matmul(D_inv, A)), D


def korenAlg(G: nx.Graph, eps = 1e-12, maxit=1e5, ndim = 2, weight = None):
    M, D = _korenMatrix(G, weight)
    n = M.shape[0]

    u = [None for i in range(ndim+1)]
    u[0] = np.ones(n)/n


    for k in range(1, ndim+1):
        v = _normalize(np.random.rand(n))
        it = 0
        while True:
            u[k] = v
            for l in range(k):
                num = np.matmul(np.matmul(u[k].T,D), u[l])
                den = np.matmul(np.matmul(u[l].T,D), u[l])
                u[k] = u[k] - (num/den)*u[l]
            
            v = _normalize(np.matmul(M, u[k]))
            it += 1

            if (np.linalg.norm(v - u[k]) < eps or it > maxit):
                break
        print(f"Eigenvector {k} computed in {it} iterations")
    
    return u

def korenTension(G: nx.Graph, n_it: int, v2: np.array, v3: np.array):   
    n = len(G.nodes)

    tension = [0 for _ in G.edges]

    for i, e in enumerate(G.edges):
        u = [np.ones(n)/n, v2.copy(), v3.copy()]
        G2 = G.copy()
        G2.remove_edges_from([e])
        if not nx.is_connected(G2):
            tension[i] = 0
            continue
        M, D = _korenMatrix(G2)
        for k in range(1, 3):
            v = u[k]
            it = 0
            for _ in range(n_it):
                u[k] = v
                for l in range(k):
                    num = np.matmul(np.matmul(u[k].T,D), u[l])
                    den = np.matmul(np.matmul(u[l].T,D), u[l])
                    u[k] = u[k] - (num/den)*u[l]
                
                v = _normalize(np.matmul(M, u[k]))
        tension[i] = np.linalg.norm(v2 - u[1]) + np.linalg.norm(v3 - u[2])
    
    return tension