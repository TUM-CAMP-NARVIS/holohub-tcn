"""
Graph visualization utility for Holoscan applications.
Analyzes the GXF graph and creates a visual representation showing operators,
their ports, and connections between them.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Any
import holoscan as hs
import logging

log = logging.getLogger(__name__)


def visualize_holoscan_graph(app: hs.core.Application, output_file: str = None, show: bool = True):
    """
    Visualize the Holoscan application graph showing operators, ports, and connections.

    Args:
        app: Holoscan Application instance
        output_file: Optional path to save the visualization (e.g., "graph.png")
        show: Whether to display the graph in a window (default: True)
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Dictionary to store operator information
    operators_info = {}
    port_nodes = []  # List of port node identifiers

    # Get the graph from the application
    graph = app.graph

    # Iterate through all operators in the graph
    for op in graph.get_nodes():
        op_name = op.name
        operators_info[op_name] = {
            'inputs': [],
            'outputs': [],
            'operator': op
        }

        # Get operator spec to access port information
        spec = op.spec

        # Collect input ports
        if hasattr(spec, 'inputs') and spec.inputs:
            for port_name, port_spec in spec.inputs.items():
                if ":" in port_name:
                    port_name, _ = port_name.split(":")
                if port_name not in operators_info[op_name]['inputs']:
                    operators_info[op_name]['inputs'].append(port_name)

        # Collect output ports
        if hasattr(spec, 'outputs') and spec.outputs:
            for port_name, port_spec in spec.outputs.items():
                operators_info[op_name]['outputs'].append(port_name)

    # Get connections from the graph
    connections = graph.get_port_connectivity_maps()[0]  # we only use the input-to-output maps

    # Build the graph
    # Add operator nodes
    for op_name in operators_info.keys():
        G.add_node(op_name, node_type='operator')

    # Add port nodes and connect them to operators
    for op_name, op_info in operators_info.items():
        # Add input port nodes
        for port_name in op_info['inputs']:
            port_id = f"{op_name}:in:{port_name}"
            G.add_node(port_id, node_type='input_port', port_name=port_name, operator=op_name)
            G.add_edge(port_id, op_name, edge_type='port_to_op')
            port_nodes.append(port_id)

        # Add output port nodes
        for port_name in op_info['outputs']:
            port_id = f"{op_name}:out:{port_name}"
            G.add_node(port_id, node_type='output_port', port_name=port_name, operator=op_name)
            G.add_edge(op_name, port_id, edge_type='op_to_port')
            port_nodes.append(port_id)

    # Add connections between ports
    for target, sources in connections.items():
        dst_op, dst_port = target.split(".")
        dst_idx = 0
        if ":" in dst_port:
            dst_port, dst_idx = dst_port.split(":")
            dst_idx = int(dst_idx)

        dst_port_id = f"{dst_op}:in:{dst_port}"
        for src in sources:
            src_op, src_port = src.split(".")

            src_port_id = f"{src_op}:out:{src_port}"

            if src_port_id in G.nodes and dst_port_id in G.nodes:
                G.add_edge(src_port_id, dst_port_id, edge_type='data_connection')
            else:
                log.warning(f"Graph: could not find source ({src_port_id}) or target ({dst_port_id})")

    # from pyvis.network import Network
    # g = Network(height=800, width=600, notebook=True)
    # g.barnes_hut()
    # g.from_nx(G)
    # g.show("graph.html")

    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 12))

    # Use hierarchical layout for better visualization
    pos = _hierarchical_layout(G, operators_info)

    # Draw edges with different styles
    data_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'data_connection']
    port_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') in ['port_to_op', 'op_to_port']]

    # Draw data connection edges (thick, colored)
    nx.draw_networkx_edges(G, pos, edgelist=data_edges, edge_color='#2E86AB',
                          width=2.5, arrows=True, arrowsize=20,
                          arrowstyle='->', ax=ax, connectionstyle='arc3,rad=0.1')

    # Draw port-to-operator edges (thin, gray)
    nx.draw_networkx_edges(G, pos, edgelist=port_edges, edge_color='#CCCCCC',
                          width=1, arrows=False, style='dashed', ax=ax)

    # Draw nodes with different colors and shapes
    operator_nodes = [n for n in G.nodes if G.nodes[n].get('node_type') == 'operator']
    input_port_nodes = [n for n in G.nodes if G.nodes[n].get('node_type') == 'input_port']
    output_port_nodes = [n for n in G.nodes if G.nodes[n].get('node_type') == 'output_port']

    # Draw operator nodes (rectangles)
    nx.draw_networkx_nodes(G, pos, nodelist=operator_nodes, node_color='#A23B72',
                          node_shape='s', node_size=3000, ax=ax)

    # Draw input port nodes (circles)
    nx.draw_networkx_nodes(G, pos, nodelist=input_port_nodes, node_color='#F18F01',
                          node_shape='o', node_size=800, ax=ax)

    # Draw output port nodes (circles)
    nx.draw_networkx_nodes(G, pos, nodelist=output_port_nodes, node_color='#C73E1D',
                          node_shape='o', node_size=800, ax=ax)

    # Draw labels
    operator_labels = {n: n for n in operator_nodes}
    port_labels = {n: G.nodes[n]['port_name'] for n in input_port_nodes + output_port_nodes}

    nx.draw_networkx_labels(G, pos, operator_labels, font_size=10,
                           font_weight='bold', font_color='black', ax=ax)
    nx.draw_networkx_labels(G, pos, port_labels, font_size=7,
                           font_color='black', ax=ax)

    # Add legend
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, fc='#A23B72', label='Operator'),
        mpatches.Circle((0.5, 0.5), 0.5, fc='#F18F01', label='Input Port'),
        mpatches.Circle((0.5, 0.5), 0.5, fc='#C73E1D', label='Output Port'),
        mpatches.FancyArrow(0, 0, 1, 0, width=0.3, color='#2E86AB', label='Data Connection'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    ax.set_title(f'Holoscan Application Graph: {app.__class__.__name__}',
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()

    # Save to file if requested
    if output_file:
        nx.write_gexf(G, output_file)
        print(f"Graph visualization saved to: {output_file}")

    # Show the plot if requested
    if show:
        plt.show()

    return G, pos


def _hierarchical_layout(G: nx.DiGraph, operators_info: Dict) -> Dict[str, Tuple[float, float]]:
    """
    Create a hierarchical layout for the graph with operators in the center
    and their ports arranged around them.

    Args:
        G: NetworkX graph
        operators_info: Dictionary containing operator and port information

    Returns:
        Dictionary mapping node IDs to (x, y) positions
    """
    pos = {}

    # First, create a simplified graph with just operators for initial layout
    op_graph = nx.DiGraph()
    for op_name in operators_info.keys():
        op_graph.add_node(op_name)

    # Add edges between operators based on data connections
    for u, v, d in G.edges(data=True):
        if d.get('edge_type') == 'data_connection':
            # Extract operator names from port IDs
            src_op = u.split(':')[0]
            dst_op = v.split(':')[0]
            if src_op in op_graph and dst_op in op_graph:
                op_graph.add_edge(src_op, dst_op)

    # Use hierarchical layout for operators
    try:
        # Try to use graphviz layout if available
        op_pos = nx.nx_agraph.graphviz_layout(op_graph, prog='dot')
    except:
        # Fallback to spring layout
        op_pos = nx.spring_layout(op_graph, k=3, iterations=50, seed=42)

    # Scale the positions
    scale_factor = 5.0
    for op in op_pos:
        op_pos[op] = (op_pos[op][0] * scale_factor, op_pos[op][1] * scale_factor)

    # Place operators
    for op_name in operators_info.keys():
        pos[op_name] = op_pos[op_name]

    # Place ports around their operators
    for op_name, op_info in operators_info.items():
        op_x, op_y = pos[op_name]

        # Place input ports to the left
        num_inputs = len(op_info['inputs'])
        for i, port_name in enumerate(op_info['inputs']):
            port_id = f"{op_name}:in:{port_name}"
            offset_y = (i - (num_inputs - 1) / 2) * 0.5
            pos[port_id] = (op_x - 1.5, op_y + offset_y)

        # Place output ports to the right
        num_outputs = len(op_info['outputs'])
        for i, port_name in enumerate(op_info['outputs']):
            port_id = f"{op_name}:out:{port_name}"
            offset_y = (i - (num_outputs - 1) / 2) * 0.5
            pos[port_id] = (op_x + 1.5, op_y + offset_y)

    return pos
