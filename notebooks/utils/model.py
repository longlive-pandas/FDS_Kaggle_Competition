from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns


def run_pca(train_df, n_components=0.95):
    """
    Performs PCA on a DataFrame and returns the transformed dataset.

    Parameters:
        train_df (pd.DataFrame): The input data (numeric only).
        n_components (float or int): Number of components or variance ratio to keep (default=0.95).

    Returns:
        pca_df (pd.DataFrame): DataFrame with principal components.
        pca_model (PCA): The trained PCA model (for later inverse_transform or analysis).
    """

    # --- Fit PCA ---
    pca_model = PCA(n_components=n_components)
    pca_values = pca_model.fit_transform(train_df)

    # --- Create new column names ---
    col_names = [f'PC{i+1}' for i in range(pca_model.n_components_)]
    pca_df = pd.DataFrame(pca_values, columns=col_names, index=train_df.index)

    # --- Print diagnostics ---
    explained = np.sum(pca_model.explained_variance_ratio_)
    print(f"✅ PCA completed.")
    print(f"Retained components: {pca_model.n_components_} / {train_df.shape[1]}")
    print(f"Total explained variance: {explained:.2f}%")
    # --- Return ---
    return pca_df, pca_model



def show_graph(train_df,player_won_col):#train_df, that you want to print and column of variable train_df['player_won']
    # Set Seaborn theme for visually appealing plots
    plot_df=pd.concat([train_df, player_won_col], axis=1)
    sns.set_theme(style="whitegrid", palette="deep", context="notebook", font_scale=1.2)
    plt.rcParams['figure.figsize'] = [10, 8]
    pair_plot = sns.pairplot(
        plot_df,
        hue='player_won',  # Color points by 'player_won' to show two groups
        diag_kind='kde',   # Kernel density plots on diagonals
        plot_kws={'alpha': 0.6, 's': 50},  # Customize scatter plot appearance
        diag_kws={'shade': True}  # Add shading to KDE for better visuals
    )
    # Add title
    pair_plot.figure.suptitle('Pair Plot of Features by Player Won', y=1.02)
    # Show plot
    plt.show()
