import matplotlib.pyplot as plt
import math
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image
import numpy as np
import seaborn as sns
import networkx as nx
from typing import Any
from typedefs import AOICenters
from kneed import KneeLocator
from matplotlib import patches


class PlotUtils:
    AOI_colors = {
        0: "#1f77b4",  
        1: "#ff7f0e",  
        2: "#2ca02c",  
        3: "#d62728",  
        4: "#9467bd",  
        5: "#8c564b",  
        6: "#e377c2",  
        7: "#7f7f7f",  
        8: "#bcbd22",  
        9: "#17becf",  
    }

    answer_colors = {
        1: '#ff0000',
        2: '#f87b63',
        3: '#ebc7b4',
        4: "#dcdfe1",
        5: '#7fb5e6',
        6: '#4187cc',
        7: '#1c5ca5',
    }

    @staticmethod
    def wrap_dict_plot(data_dict: dict[Any, pd.DataFrame], n_wrap: int, plotting_function, title: str = None, image: Image = None):
        _, axs = PlotUtils.wrap_subplots(len(data_dict), n_wrap)
        for i, (subject, trial) in enumerate(data_dict.items()):
            if image:
                plotting_function(trial, title=f"{title} {subject}", image=image, ax=axs[i])
            else:
                plotting_function(trial, title=f"{title} {subject}", ax=axs[i])


    @staticmethod
    def wrap_subplots(n_plots: int, n_wrap: int) -> tuple[Figure, list[Axes]]:
        ncols = min(n_plots, n_wrap)
        nrows = math.ceil(n_plots / n_wrap)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 5))
        if n_plots == 1:
            axs = np.array([axs])
        axs: list[Axes] = list(axs.flatten())
        for ax in axs[n_plots:]:
            ax.axis("off")
        return fig, axs


    @staticmethod
    def trace_plot(
        df: pd.DataFrame, 
        title: str = None,
        color_marker: str = "red",
        color_arrow: str = "white",
        image: Image = None, 
        ax: Axes = None
    ):
        if not ax:
            _, ax = plt.subplots(1, 1)
        if image:
            ax.imshow(image)
        if title:
            ax.set(title=title)

        for i in range(len(df.index) - 1):
            ax.annotate(
                "", 
                xy=(df["x"][i + 1], df["y"][i + 1]), 
                xytext=(df["x"][i], df["y"][i]), 
                arrowprops=dict(arrowstyle="->", color=color_arrow, lw=0.5),
            )
        ax.scatter(df["x"], df["y"], color=color_marker, marker=".")

    
    @staticmethod
    def clustering_plot(df: pd.DataFrame, title: str = None, image: Image = None, ax: Axes = None):
        if not ax:
            _, ax = plt.subplots(1, 1)
        if image:
            ax.imshow(image)
        if title:
            ax.set(title=title)
        sns.scatterplot(data=df, x='x', y='y', hue='AOI', palette='deep', s=80, ax=ax)


    @staticmethod
    def elbow_plot(inertias: dict[int, float]) -> None:
        k_values, inertia_values = zip(*inertias.items())
        knee_locator = KneeLocator(k_values, inertia_values, curve='convex', direction='decreasing')
        k = knee_locator.knee
        _, ax = plt.subplots(1, 1, figsize=(6.5, 4))
        ax.plot(k_values, inertia_values, marker='o')
        ax.axvline(x=k, color="red")
        ax.set(xlabel='Number of clusters [-]', ylabel='Inertia [-]', title=f'Elbow method for optimal k')
        ax.set_xticks(k_values)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))


    @staticmethod
    def plot_aois(aois: AOICenters, title: str = None, image: Image = None, ax: Axes = None):
        
        if not ax:
            _, ax = plt.subplots(1, 1)
        if image:
            ax.imshow(image)
        if title:
            ax.set(title=title)
        for aoi, (x, y) in aois.items():
            color = PlotUtils.AOI_colors.get(aoi % 10, "#696969")
            ax.scatter(x, y, s=200, edgecolors="white", color=color)
            ax.text(x + 1, y + 5, str(aoi), fontsize=12, ha='center', va='center', color='white' )

    
    @staticmethod 
    def divide_aoi_plot(image: Image, ax: Axes = None):
        if not ax:
            _, ax = plt.subplots(1, 1)

        width, height = image.size
        rects = [
            (0, 0, width/2, height/2),       
            (width/2, 0, width/2, height/2),
            (0, height/2, width/2, height/2),
            (width/2, height/2, width/2, height/2)
        ]

        ax.imshow(image)
        for i, (x, y, w, h) in enumerate(rects):
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor='none',
                facecolor=PlotUtils.AOI_colors[i], alpha=0.3
            )
            ax.add_patch(rect)


    @staticmethod
    def heatmap(transition_matrix: np.ndarray, title: str = None, ax: Axes = None):
        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        if title:
            ax.set(title=title)
        sns.heatmap(transition_matrix, fmt='.1f', cmap='coolwarm', annot=True, ax=ax)
        

    @staticmethod
    def draw(G: nx.Graph, weights: bool = True, title: str = None, ax=None) -> None:
        import matplotlib.pyplot as plt
        import networkx as nx

        if not ax:
            _, ax = plt.subplots(1, 1)
        if title:
            ax.set(title=title)

        pos = nx.forceatlas2_layout(G, seed=44)
        edge_curve = "arc3,rad=0.1"
        colors = [PlotUtils.AOI_colors.get(node % 10, "#7f7f7f") for node in G.nodes]

        nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color=colors)
        nx.draw_networkx_labels(G, pos=pos, ax=ax)
        nx.draw_networkx_edges(G, pos=pos, ax=ax, connectionstyle=edge_curve, arrows=True, arrowsize=12)

        if weights:
            edge_labels = {
                edge: f"{int(w)}" if float(w).is_integer() else f"{w:.1f}"
                for edge, w in nx.get_edge_attributes(G, "weight").items()
            }
            nx.draw_networkx_edge_labels(
                G=G,
                pos=pos,
                edge_labels=edge_labels,
                ax=ax,
                label_pos=0.6,
                rotate=False,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=1),
            )

    
    @staticmethod
    def centrality_bar_plot(centrality_dict: dict[int, np.ndarray], title: str, ax: Axes = None):
        if not ax:
            _, ax = plt.subplots(1, 1)
        stack = np.vstack([list(centrality_dict.values())])
        means = np.mean(stack, axis=0)
        stds = np.std(stack, axis=0)
        aois = np.arange(len(means))
        colors = [PlotUtils.AOI_colors.get(aoi % 10, "#7f7f7f") for aoi in aois]

        ax.bar(aois, means, yerr=stds, capsize=5, color=colors)
        ax.set_xticks(aois, [f'AOI {i}' for i in aois])
        ax.set_ylabel(f"{title}")
        ax.set_title(f'{title} means for each AOI across subjects')


    @staticmethod
    def dimensionality_reduction_plot(reduced_data: np.ndarray, labels: dict[int, int], title: str = None, annotate: bool = False, ax: Axes = None):
        if not ax:
            _, ax = plt.subplots(1, 1)
        if title:
            ax.set(title=title)

        scatter = ax.scatter(
            reduced_data[:, 0], reduced_data[:, 1],
            c=[PlotUtils.answer_colors[x] for x in labels.values()],              
            s=250,                 
            edgecolor='k'
        )
        if annotate:
            for i, label in enumerate(labels.keys()):
                ax.text(
                    reduced_data[i,0], reduced_data[i,1], str(label),
                    color='black', fontsize=9,
                    ha='center', va='center'
                )
            
        box = ax.get_position()  
        ax.set_position([box.x0, box.y0, box.width*0.85, box.height])  

        patches_list = [patches.Patch(color=color, label=str(label)) for label, color in PlotUtils.answer_colors.items()]
        ax.legend(handles=patches_list, title="Answer", loc='center left', bbox_to_anchor=(1, 0.5))