import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_complete_grouped_reliability_plot():
    # Create DataFrame with all network reliability data
    data = {
        'Network': [
            # DMN Network
            'LP (L)', 'LP (R)', 'MPFC', 'PCC',
            # SN Network
            'ACC', 'AInsula (L)', 'AInsula (R)', 
            'RPFC (L)', 'RPFC (R)', 'SMG (L)', 'SMG (R)',
            # FPN Network
            'LPFC (L)', 'LPFC (R)',
            'PPC (L)', 'PPC (R)',
            # Dorsal Attention Network
            'FEF (L)', 'FEF (R)', 
            'IPS (L)', 'IPS (R)',
            # Visual Network
            'Lateral visual (L)', 'Lateral visual (R)', 'Medial visual', 'Occipital visual',
            # Language Network
            'IFG (L)', 'IFG (R)', 'pSTG (L)', 'pSTG (R)',
            # Sensorimotor Network
            'Lateral (L)', 'Lateral (R)', 'Superior',
            # Cerebellar Network
            'Cerebellar Anterior', 'Cerebellar Posterior'
        ],
        'Network_Group': [
            # Network groupings
            'DMN', 'DMN', 'DMN', 'DMN',
            'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN',
            'FPN', 'FPN', 'FPN', 'FPN',
            'DAN', 'DAN', 'DAN', 'DAN',
            'VN', 'VN', 'VN', 'VN',
            'LN', 'LN', 'LN', 'LN',
            'SMN', 'SMN', 'SMN',
            'Cerebellar', 'Cerebellar'
        ],
        'All_Timepoints': [
            0.727, 0.616, 0.478, 0.529,
            0.595, 0.581, 0.613, 0.575, 0.632, 0.688, 0.687,
            0.519, 0.583, 0.602, 0.593,
            0.286, 0.288, 0.591, 0.628,
            0.521, 0.496, 0.529, 0.360,
            0.551, 0.614, 0.603, 0.586,
            0.496, 0.564, 0.538,
            0.452, 0.516
        ],
        'Baseline_1h': [
            0.632, 0.491, 0.368, 0.479,
            0.505, 0.459, 0.518, 0.555, 0.588, 0.578, 0.651,
            0.427, 0.460, 0.529, 0.479,
            0.134, 0.254, 0.533, 0.587,
            0.481, 0.464, 0.434, 0.308,
            0.447, 0.492, 0.502, 0.419,
            0.365, 0.429, 0.327,
            0.291, 0.456
        ],
        'Baseline_1m': [
            0.627, 0.521, 0.379, 0.476,
            0.488, 0.425, 0.487, 0.424, 0.510, 0.618, 0.551,
            0.384, 0.454, 0.497, 0.501,
            0.263, 0.120, 0.424, 0.499,
            0.371, 0.345, 0.454, 0.329,
            0.538, 0.555, 0.567, 0.557,
            0.389, 0.474, 0.492,
            0.239, 0.403
        ],
        'Hour1_1m': [
            0.670, 0.561, 0.355, 0.325,
            0.500, 0.557, 0.560, 0.442, 0.518, 0.614, 0.617,
            0.447, 0.523, 0.501, 0.505,
            0.169, 0.225, 0.503, 0.500,
            0.396, 0.353, 0.378, 0.184,
            0.389, 0.511, 0.469, 0.509,
            0.413, 0.475, 0.466,
            0.490, 0.350
        ]
    }
    
    df = pd.DataFrame(data)
    df_melted = df.melt(id_vars=['Network', 'Network_Group'], 
                       var_name='Comparison', 
                       value_name='ICC')
    
    # Create the plot
    plt.figure(figsize=(15, 20))
    
    # Create horizontal bar plot
    ax = sns.barplot(data=df_melted, 
                    y='Network', 
                    x='ICC',
                    hue='Comparison',
                    palette=['#8884d8', '#82ca9d', '#ffc658', '#ff7300'])
    
        # Increase font sizes
    plt.rcParams.update({'font.size': 14})  # Increase base font size
    
    # Network group labels with larger font
    current_pos = 0
    for group in df.Network_Group.unique():
        group_size = len(df[df.Network_Group == group])
        plt.axhline(y=current_pos - 0.5, color='gray', linestyle='-', alpha=0.3)
        plt.text(-0.2, current_pos + (group_size/2) - 0.5, group, 
                rotation=90, verticalalignment='center', fontsize=14, fontweight='bold')
        current_pos += group_size
    
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, zorder=0)
    
    # Increase title and axis labels font size
    plt.title('Network ICC reliability across time points', 
             pad=20, fontsize=16)
    plt.xlabel('ICC Value', fontsize=14)
    plt.ylabel('Brain Regions', fontsize=14)
    
    # Increase tick labels font size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Increase legend font size
    plt.legend(fontsize=14)
    
    plt.tight_layout()
    return plt

# Create and save the plot
plot = create_complete_grouped_reliability_plot()
plot.savefig('/output/network_reliability_grouped_complete.png', dpi=300, bbox_inches='tight')
plot.savefig('/output/network_reliability_grouped_complete.svg', format='svg', bbox_inches='tight')