import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

def prepare_data(file_path):
    df = pd.read_excel(file_path)
    clinical_vars = df.columns[2:].tolist()
    
    # Get unique brain regions and their networks
    regions_with_networks = set()
    for _, row in df.iterrows():
        parts1 = row['Region 1'].strip().split(' ', 1)  # Split only at the first space
        parts2 = row['Region 2'].strip().split(' ', 1)  # Split only at the first space
        regions_with_networks.add(tuple(parts1))
        regions_with_networks.add(tuple(parts2))
    
    return df, clinical_vars, list(regions_with_networks)

def get_network_colors():
    # Define colors for each network
    network_colors = {
        'DMN': '#E74C3C',    # Red
        'SN': '#2ECC71',     # Green
        'LN': '#3498DB',     # Blue
        'FPN': '#9B59B6',    # Purple
        'DAN': '#F1C40F',    # Yellow
        'SMN': '#95A5A6'     # Gray
    }
    return network_colors

def create_fixed_layout(regions_with_networks):
    # Group regions by networks
    networks = {}
    for network, region in regions_with_networks:
        if network not in networks:
            networks[network] = []
        networks[network].append((network, region))
    
    # Create a layout where regions of the same network are grouped
    pos = {}
    
    # Define positions for each network (original version)
    network_positions = {
        'LN': (0, 0),        # Center
        'DMN': (-2.5, 1.5),  # Left up
        'FPN': (2.5, 1.5),   # Right up
        'SN': (0, 3),        # Top center
        'DAN': (-2.5, -1.5), # Left down
        'SMN': (2.5, -1.5)   # Right down
    }

    # For each network, distribute regions in a circle with radius based on number of regions
    for network, regions in networks.items():
        center_x, center_y = network_positions.get(network, (0, 0))
        
        # Calculate radius based on number of regions
        num_regions = len(regions)
        base_radius = 0.4  # Base radius
        radius = base_radius * (1 + (num_regions / 5))  # Increases with more regions
        
        # Distribute regions in a circle
        for i, region_tuple in enumerate(regions):
            angle = 2 * np.pi * i / num_regions
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            pos[region_tuple] = (x, y)
    
    return pos

def create_clinical_network_visualizations(df, clinical_vars, fixed_pos, all_regions):
    n_vars = len(clinical_vars)
    n_cols = 1
    n_rows = 2

    network_colors = get_network_colors()

    fig = plt.figure(figsize=(24, 8*n_rows))  # Larger figure
    fig.suptitle('Brain Network Correlations by Clinical Variable', fontsize=24, y=1.03, fontweight='bold')
    plt.subplots_adjust(top=0.80, hspace=0.6, wspace=0.3) 

    # Determine which is the last subplot
    last_subplot_idx = len(clinical_vars) - 1

    for idx, var in enumerate(clinical_vars):
        ax = plt.subplot(n_rows, n_cols, idx + 1)

        G = nx.Graph()

        # Add ALL nodes first (to keep positions consistent)
        for region in all_regions:
            G.add_node(region)

        # Determine which have significant correlations for this var
        active_edges = []
        for _, row in df.iterrows():
            if pd.notnull(row[var]):
                correlation = float(row[var])
                if abs(correlation) >= 0.5:
                    parts1 = row['Region 1'].strip().split(' ', 1)
                    parts2 = row['Region 2'].strip().split(' ', 1)
                    region1 = tuple(parts1)
                    region2 = tuple(parts2)
                    active_edges.append((region1, region2, correlation))

        # Extract active regions (with significant correlations)
        active_regions = set()
        for r1, r2, _ in active_edges:
            active_regions.add(r1)
            active_regions.add(r2)

        plt.title(f'{var}', fontsize=20, fontweight='bold')  # Larger title

        # Add edges
        for r1, r2, corr in active_edges:
            G.add_edge(r1, r2, weight=corr)

        # Get node colors
        node_colors = []
        for node in G.nodes():
            if node in active_regions:
                # Active nodes with their network color
                node_colors.append(network_colors.get(node[0], 'gray'))
            else:
                # Inactive nodes in light gray and transparent
                node_colors.append('lightgray')

        # Edge colors and thickness
        edge_colors = ['red' if G[u][v]['weight'] < 0 else 'blue' for u, v in G.edges()]
        edge_weights = [abs(G[u][v]['weight']) * 2 for u, v in G.edges()]

        # Draw the graph
        nx.draw_networkx_nodes(G, fixed_pos, 
                             node_size=[1000 if node in active_regions else 400 for node in G.nodes()],
                             node_color=node_colors, 
                             alpha=[0.9 if node in active_regions else 0.3 for node in G.nodes()])

        nx.draw_networkx_edges(G, fixed_pos, width=edge_weights, 
                             edge_color=edge_colors, alpha=0.85)

        # Only labels for active nodes
        labels = {node: node[1] for node in G.nodes() if node in active_regions}
        nx.draw_networkx_labels(G, fixed_pos, labels, font_size=16, font_weight='bold')  # Larger and bold labels

        # Add legend only in the last subplot
        if idx == last_subplot_idx:
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, label=network,
                                        markersize=18)
                             for network, color in network_colors.items()]
            # Add legend for edge colors
            legend_elements.extend([
                plt.Line2D([0], [0], color='blue', label='Positive Correlation', linewidth=3),
                plt.Line2D([0], [0], color='red', label='Negative Correlation', linewidth=3)
            ])
            ax.legend(handles=legend_elements, loc='upper center', 
                     bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=16, frameon=True)

        plt.axis('off')

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Load and prepare data
    file_path = '/input/rm_correlation_matrix_formatted.xlsx'
    df, clinical_vars, all_regions = prepare_data(file_path)
    
    # Create fixed layout with ALL regions
    fixed_pos = create_fixed_layout(all_regions)
    
    # Create and save network visualization
    network_fig = create_clinical_network_visualizations(df, clinical_vars, fixed_pos, all_regions)
    
    # Save in both formats
    network_fig.savefig('/output/clinical_networks_overview.png', 
                       dpi=300, bbox_inches='tight')
    network_fig.savefig('/output/clinical_networks_overview.svg', 
                       format='svg', bbox_inches='tight')