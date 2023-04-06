import torch
import copy
from detectron2.structures import BoxMode, Boxes, Instances, pairwise_iou

 
def ncut_cost_torch(cut, D, W):
    """Returns the N-cut cost of a bi-partition of a graph.

    Parameters
    ----------
    cut : ndarray
        The mask for the nodes in the graph. Nodes corresponding to a `True`
        value are in one set.
    D : csc_matrix
        The diagonal matrix of the graph.
    W : csc_matrix
        The weight matrix of the graph.

    Returns
    -------
    cost : float
        The cost of performing the N-cut.

    References
    ----------
    .. [1] Normalized Cuts and Image Segmentation, Jianbo Shi and
           Jitendra Malik, IEEE Transactions on Pattern Analysis and Machine
           Intelligence, Page 889, Equation 2.
    """
    num = cut.shape[0]
    cut_cost = ((cut.unsqueeze(1).expand(num, num) ^ cut) * W).sum() / 2
    # D has elements only along the diagonal, one per node, so we can directly
    # index the data attribute with cut.
    assoc_a = D[cut].sum()
    assoc_b = D[~cut].sum()

    return (cut_cost / assoc_a) + (cut_cost / assoc_b)

def get_min_ncut_torch(ev, d, w, num_cuts):
    mcut = torch.inf
    mn = ev.min()
    mx = ev.max()

    # If all values in `ev` are equal, it implies that the graph can't be
    # further sub-divided. In this case the bi-partition is the the graph
    # itself and an empty set.
    min_mask = torch.zeros_like(ev, dtype=bool)
    if torch.allclose(mn, mx):
        return min_mask, mcut

    # Refer Shi & Malik 2001, Section 3.1.3, Page 892
    # Perform evenly spaced n-cuts and determine the optimal one.
    for t in torch.linspace(mn, mx-((mx - mn) / num_cuts), num_cuts):
        mask = ev > t
        cost = ncut_cost_torch(mask, d, w)
        if cost < mcut:
            min_mask = mask
            mcut = cost

    return min_mask, mcut

def _label_all_torch(subgraph_nodes, original_labels, new_labels):
    node = min(subgraph_nodes)
    new_label = original_labels[node]
    new_labels[subgraph_nodes] = new_label

def _ncut_relabel(w, subgraph_nodes, thresh, num_cuts, original_labels, new_labels):
    d = torch.diag(w.sum(0))
    m = w.shape[0]
    if m > 2:
        d2 = copy.deepcopy(d)
        # Since d is diagonal, we can directly operate on its data
        # the inverse of the square root
        d2 = torch.diag(torch.reciprocal(torch.sqrt(torch.diag(d2))))
        # Refer Shi & Malik 2001, Equation 7, Page 891
        A = torch.matmul(torch.matmul(d2, (d - w)), d2)
        # v0 = torch.rand(A.shape[0])
        # vals, vectors = linalg.eigsh(A.cpu().numpy(), which='SM', v0=v0, k=min(100, m - 2))
        # vals, vectors = np.real(vals), np.real(vectors)
        # index2 = _ncut_cy.argmin2(vals)
        # ev = torch.from_numpy(vectors[:, index2]).to(w.device)

        k=min(100, m - 2)
        vals, vectors = torch.linalg.eig(A)
        vals, vectors = torch.real(vals), torch.real(vectors)
        vals, index = vals.sort()
        # vals = vals[:k]
        vectors = vectors[:, index[:k]]
        index2 = 1 if k >= 2 else 0
        ev = vectors[:, index2]
        
        cut_mask, mcut = get_min_ncut_torch(ev, d, w, num_cuts)
        if (mcut < thresh):
            sub1 = torch.where(cut_mask == True)[0]
            sub2 = torch.where(cut_mask != True)[0]
            _ncut_relabel(w[sub1][:, sub1], subgraph_nodes[sub1], thresh, num_cuts, original_labels, new_labels)
            _ncut_relabel(w[sub2][:, sub2], subgraph_nodes[sub2], thresh, num_cuts, original_labels, new_labels)
            return

    _label_all_torch(subgraph_nodes, original_labels, new_labels)


def torch_ncut(A, original_labels, new_labels, thresh=0.1, num_cuts=10, device=torch.device('cpu')):
    w = A.to(device) + torch.eye(A.shape[0], device=device)
    node_index = torch.range(0, A.shape[0]-1, dtype=torch.int64, device=device)
    _ncut_relabel(w, node_index, thresh, num_cuts, original_labels, new_labels)
    return new_labels

def pairwise_function(gtRects, detRects, device=torch.device('cpu')):
    gtRects = gtRects.to(device)
    detRects = detRects.to(device)
    detRects[:, 2] = detRects[:, 0] + detRects[:, 2]
    detRects[:, 3] = detRects[:, 1] + detRects[:, 3]
    gtRects[:, 2] = gtRects[:, 0] + gtRects[:, 2]
    gtRects[:, 3] = gtRects[:, 1] + gtRects[:, 3]
    
    iou_matrix = pairwise_iou(Boxes(gtRects), Boxes(detRects))
    return iou_matrix

def torch_ncut_detection(proposals, sim_matrix=None, original_labels=None, thresh=0.1, num_cuts=10, device=torch.device('cpu')):
    w = torch.tensor(pairwise_function(proposals, proposals, device), dtype=torch.float64)
    if sim_matrix is not None:
        w = w * torch.nn.Sigmoid()(sim_matrix).to(device)
    # w = A + torch.eye(A.shape[0], device=device)
    node_index = torch.range(0, w.shape[0]-1, dtype=torch.int64, device=device)
    if original_labels == None:
        original_labels = torch.range(0, w.shape[0]-1, dtype=torch.int64, device=device)
    new_labels = torch.zeros(w.shape[0], dtype=torch.int64, device=device)
    _ncut_relabel(w, node_index, thresh, num_cuts, original_labels, new_labels)
    return new_labels.cpu()
