from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch
import torch.nn.functional as F
import torch.nn as nn
import torch

class GraphStateUpdaterGNN(nn.Module):
    """
    Graph Neural Network module to update and represent the state of a graph.
    It computes a graph-level embedding from the input graph structure.
    """
    def __init__(self, node_embedding_dim: int, gnn_hidden_dim: int, output_dim: int):
        """
        Initializes the GraphStateUpdaterGNN.

        Args:
            node_embedding_dim (int): Dimensionality of the learnable initial node embeddings.
            gnn_hidden_dim (int): Hidden dimensionality of the GCN layers.
            output_dim (int): Dimensionality of the final output graph embedding.
        """
        super().__init__()

        # A single learnable parameter vector to initialize features for all nodes.
        # This vector is cloned for each node in the input graph.
        self.initial_node_emb = nn.Parameter(torch.randn(1, node_embedding_dim))

        # First Graph Convolutional Network layer.
        # It takes initial node embeddings and transforms them based on graph connectivity.
        # `bias=False` is an architectural choice.
        self.conv1 = GCNConv(node_embedding_dim, gnn_hidden_dim, bias=False)

        # Second Graph Convolutional Network layer, further refining node representations.
        self.conv2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim, bias=False)

        # A linear layer to project the pooled graph-level features to the final output dimension.
        self.output_projection = nn.Linear(gnn_hidden_dim, output_dim)

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Processes a batch of graphs (or a single graph wrapped in a Batch object)
        to compute their graph-level embeddings.

        Args:
            data (torch_geometric.data.Batch): A batch of graph data.
                Expected attributes:
                - data.num_nodes: Total number of nodes in the batch.
                - data.edge_index: Edge connectivity information.
                - data.batch: Tensor mapping each node to its respective graph in the batch.
                              (data.edge_attr is ignored as per design).

        Returns:
            torch.Tensor: Graph-level embeddings for each graph in the batch.
                          Shape: [num_graphs_in_batch, output_dim]
        """

        # Handle cases where an empty graph (0 nodes) might be passed.
        if data.num_nodes == 0:
            # If there are no nodes, create an empty tensor for initial features.
            # The subsequent check `if x.shape[0] == 0:` will handle returning zeros.
            x = torch.empty((0, self.initial_node_emb.shape[1]), device=self.initial_node_emb.device)
        else:
            # Create initial node features by repeating the learnable embedding for each node.
            x = self.initial_node_emb.repeat(data.num_nodes, 1)

        # If, after initialization, x has no nodes (e.g. num_nodes was 0),
        # return a zero tensor of the correct output shape.
        # This prevents errors in GCN layers or pooling with empty inputs.
        if x.shape[0] == 0:
            return torch.zeros((data.num_graphs, self.output_projection.out_features), device=x.device)

        # Pass node features and connectivity through the first GCN layer.
        # edge_weight is explicitly set to None as we only care about graph structure.
        x_conv1 = self.conv1(x, data.edge_index, edge_weight=None)
        x_conv1_relu = F.relu(x_conv1) # Apply ReLU activation

        # Check for NaNs after the first convolution (useful for debugging).
        if x_conv1.isnan().any():
            # If NaNs are produced, return a NaN tensor to signal an issue.
            nan_output = torch.full((data.num_graphs, self.output_projection.out_features), float('nan'), device=x.device)
            return nan_output

        # Pass the features through the second GCN layer.
        x_conv2 = self.conv2(x_conv1_relu, data.edge_index, edge_weight=None)
        x_conv2_relu = F.relu(x_conv2) # Apply ReLU activation

        # If node features become empty after convolutions (e.g., if all nodes were filtered out, though unlikely here),
        # return a zero tensor before pooling.
        if x_conv2_relu.shape[0] == 0 :
            return torch.zeros((data.num_graphs, self.output_projection.out_features), device=x_conv2_relu.device)

        # Aggregate node features into graph-level embeddings using mean pooling.
        # `data.batch` ensures pooling is done per graph in the batch.
        graph_embedding_pooled = global_mean_pool(x_conv2_relu, data.batch)

        # Project the pooled embeddings to the final output dimension.
        graph_embedding_projected = self.output_projection(graph_embedding_pooled)

        return graph_embedding_projected

class AutoregressiveGraphGenerator(nn.Module):
    """
    Autoregressive model for generating graphs node by node, and edge by edge.
    It uses a GraphStateUpdaterGNN to process the current graph state and
    predicts connections for new nodes to existing ones.
    """
    def __init__(self, graph_updater: GraphStateUpdaterGNN, graph_embedding_dim: int, max_nodes_to_generate: int):
        """
        Initializes the AutoregressiveGraphGenerator.

        Args:
            graph_updater (GraphStateUpdaterGNN): An instance of the GNN module used to get graph state embeddings.
            graph_embedding_dim (int): The dimensionality of the embeddings produced by the graph_updater.
                                       This serves as the input dimension for the edge prediction head.
            max_nodes_to_generate (int): The maximum number of nodes the model is designed to predict connections for.
                                         This determines the output size of the edge prediction linear layer.
        """
        super().__init__()
        self.graph_updater = graph_updater # The GNN module to get graph state
        self.graph_embedding_dim = graph_embedding_dim
        self.max_nodes_to_generate = max_nodes_to_generate

        # A linear layer to predict edge logits.
        # Input: graph state embedding.
        # Output: logits for connecting the current new node to all possible previous node positions
        # (up to max_nodes_to_generate).
        self.edge_prediction_head = nn.Linear(graph_embedding_dim, max_nodes_to_generate)

    def forward(self, data: Batch, current_node_idx_in_graph: int) -> torch.Tensor:
        """
        Performs one step of the autoregressive generation/training process.
        Given the current graph state and the index of the node being added/processed,
        it predicts the logits for connecting this node to all previously existing nodes.

        Args:
            data (Batch): A PyG Batch object representing the current state of the graph(s).
                          For true autoregressive training, this is the graph built so far.
            current_node_idx_in_graph (int): The 0-indexed ID of the node for which
                                             we are predicting edges to *previous* nodes (0 to current_node_idx_in_graph - 1).

        Returns:
            torch.Tensor: A tensor of edge logits.
                          Shape: [num_graphs_in_batch, current_node_idx_in_graph].
                          Each logit corresponds to a potential edge from the current node
                          to a previous node.
        """
        # The first node (index 0) has no previous nodes to connect to.
        if current_node_idx_in_graph == 0:
            num_graphs = data.num_graphs if data is not None else 1 # Handle potential None data if called with 0
            # Determine device, fall back to CPU if data or edge_index is not available for device inference
            device_to_use = 'cpu'
            if data is not None and hasattr(data, 'edge_index') and data.edge_index is not None:
                device_to_use = data.edge_index.device
            elif hasattr(self.edge_prediction_head.weight, 'device'):
                 device_to_use = self.edge_prediction_head.weight.device

            return torch.zeros((num_graphs, 0), device=device_to_use) # Return shape [num_graphs, 0]

        # Obtain the graph state embedding from the graph_updater GNN.
        graph_state_embedding = self.graph_updater(data)

        # Pass the graph state embedding through the edge prediction head.
        # This produces logits for connecting to ALL possible previous node positions (up to max_nodes_to_generate).
        edge_logits_all_possible_previous = self.edge_prediction_head(graph_state_embedding)

        # Slice the logits to get only those relevant to the actually existing previous nodes.
        # We need predictions for connections to nodes 0, 1, ..., current_node_idx_in_graph - 1.
        # The number of such nodes is current_node_idx_in_graph.
        relevant_edge_logits = edge_logits_all_possible_previous[:, :current_node_idx_in_graph]

        return relevant_edge_logits