from torch_geometric.utils import coalesce
from torch_geometric.data import Batch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import scipy.sparse
import numpy as np
import torch

@torch.no_grad()
def generate_graph(model, num_nodes_to_generate, device, sampling_threshold=0.5):
    model.eval()
    if num_nodes_to_generate <= 0:
        return Data(num_nodes=0, edge_index=torch.empty((2,0), dtype=torch.long))
    if num_nodes_to_generate == 1:
        return Data(num_nodes=1, edge_index=torch.empty((2,0), dtype=torch.long))
    generated_edges_list_tuples = set()
    current_num_nodes = 1
    for new_node_idx in range(1, num_nodes_to_generate):
        if not generated_edges_list_tuples:
            current_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            edge_list_for_tensor = []
            for u, v in generated_edges_list_tuples:
                edge_list_for_tensor.append([u,v])
                edge_list_for_tensor.append([v,u])
            if not edge_list_for_tensor:
                current_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            else:
                current_edge_index = torch.tensor(edge_list_for_tensor, dtype=torch.long, device=device).t().contiguous()
                current_edge_index = coalesce(current_edge_index)
        current_graph_data = Data(num_nodes=current_num_nodes, edge_index=current_edge_index).to(device)
        current_graph_as_batch = Batch.from_data_list([current_graph_data])
        edge_logits = model(current_graph_as_batch, new_node_idx)
        edge_probs = torch.sigmoid(edge_logits.squeeze(0))
        edge_decisions = torch.bernoulli(edge_probs)
        for prev_node_idx in range(new_node_idx):
            if edge_probs[prev_node_idx] > 0.3:
                generated_edges_list_tuples.add((new_node_idx, prev_node_idx))
                generated_edges_list_tuples.add((prev_node_idx, new_node_idx))
        current_num_nodes += 1
    if not generated_edges_list_tuples:
        final_edge_index = torch.empty((2,0), dtype=torch.long)
    else:
        final_edge_list_symmetric = []
        for u,v in generated_edges_list_tuples:
            final_edge_list_symmetric.append([u,v])
            final_edge_list_symmetric.append([v,u])
        if not final_edge_list_symmetric:
            final_edge_index = torch.empty((2,0), dtype=torch.long)
        else:
            final_edge_index = torch.tensor(final_edge_list_symmetric, dtype=torch.long).t().contiguous()
            final_edge_index = coalesce(final_edge_index)
    generated_graph = Data(num_nodes=num_nodes_to_generate, edge_index=final_edge_index.to('cpu'))
    return generated_graph

def pyg_graph_to_scipy_sparse(graph_data, value_to_fill=1.0):
    num_nodes = graph_data.num_nodes
    if graph_data.edge_index is None or graph_data.edge_index.numel() == 0:
        return scipy.sparse.csr_matrix((num_nodes, num_nodes), dtype=np.float64 if isinstance(value_to_fill, float) else type(value_to_fill))
    edge_index = graph_data.edge_index.cpu().numpy()
    rows = edge_index[0]
    cols = edge_index[1]
    data_values = np.full(rows.shape[0], value_to_fill, dtype=type(value_to_fill))
    sparse_matrix = scipy.sparse.coo_matrix((data_values, (rows, cols)), shape=(num_nodes, num_nodes))
    return sparse_matrix.tocsr()