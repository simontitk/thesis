import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import seaborn as sns
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import LeaveOneOut
from dataloader import DataLoader
from PIL import Image
from _graphutils import GraphUtils
from _plotutils import PlotUtils
from kernels.wlkernel import WLKernel
from kernels.shortestpathkernel import ShortestPathKernel
from kernels.randomwalkkernel import RandomWalkKernel
from kernels.labelhasher import LabelHasher
from sklearn.manifold import MDS, TSNE


class Pipeline:

    def shortest_path_kernel_clustering(img: str, PLOT: bool = False):
        dl = DataLoader("../eyelink_data")
        img_file = dl.get_image(img)

        trials = dl.get_image_trials(img)
        trial_numbers = dl.get_image_trials(img, False)
        answer_dfs = dl.get_image_answers(img)
        answers = {subject: answer_dfs[subject].query(f"trial == {trial_numbers[subject]}")["response"].item() for subject in trials.keys()}

        if PLOT:
            PlotUtils.wrap_dict_plot(trials, n_wrap=3, plotting_function=PlotUtils.trace_plot, title="Subject", image=img_file)



        # CLUSTERING 
        MIN_CLUSTERS = 2
        MAX_CLUSTERS = 10

        trials_clustered: dict[int, pd.DataFrame] = {}
        cluster_centers: dict[int, dict[int, tuple[float, float]]] = {}
        for (subject, trial) in trials.items():
            df_clustered, intertia, cluster_c, k = GraphUtils.optimal_clustering(MIN_CLUSTERS, MAX_CLUSTERS, trial)
            trials_clustered[subject] = df_clustered
            cluster_centers[subject] = cluster_c

        if PLOT:
            PlotUtils.wrap_dict_plot(trials_clustered, n_wrap=3, plotting_function=PlotUtils.clustering_plot, title="Subject", image=img_file)




        # CLUSTER CENTERS
        if PLOT:
            _, axs = PlotUtils.wrap_subplots(len(cluster_centers), 2)
            for i, (subject, item) in enumerate(cluster_centers.items()):
                for aoi, (x, y) in item.items():
                    axs[i].set(title=f"Centers for subject {subject}")
                    axs[i].imshow(img_file)
                    axs[i].scatter(x, y, s=200)
                    axs[i].text(x + 5, y + 5, str(aoi), fontsize=12, ha='center', va='center', color='white')




        # ALL CLUSTER CENTERS IN ONE 
        _, ax= plt.subplots(1,1)
        for i, (subject, item) in enumerate(cluster_centers.items()):
            for aoi, (x, y) in item.items():
                ax.imshow(img_file)
                ax.scatter(x, y, s=200)
                ax.text(x + 5, y + 5, str(aoi), fontsize=12, ha='center', va='center', color='white')
        ax.set(title=f"Centers for all subjects")




        # TRANSITION MATRCIES
        matrices = {subject: GraphUtils.transition_matrix(trial) for (subject, trial) in trials_clustered.items()}
        if PLOT:
            PlotUtils.wrap_dict_plot(matrices, n_wrap=3, plotting_function=PlotUtils.heatmap, title="Subject")



        # GRAPHS
        graphs = {subject: GraphUtils.transition_matrix_to_graph(matrix) for (subject, matrix) in matrices.items()}
        if PLOT:
            PlotUtils.wrap_dict_plot(graphs, n_wrap=3, plotting_function=PlotUtils.draw, title="Subject")




        # PRECOMPUTED KERNEL MATRIX
        subjects_order =list(matrices.keys())
        K_matrix = np.zeros(shape=(len(matrices), len(matrices)))
        for i, (subject1, matrix1) in enumerate(matrices.items()):
            for j, (subject2, matrix2) in enumerate(matrices.items()):
                K = ShortestPathKernel.get_kernel_value(matrix1, matrix2, kernel=ShortestPathKernel.inverse_abs_diff_kernel, normalize=True)
                K_matrix[i][j] = K
        if PLOT:
            PlotUtils.heatmap(K_matrix)


        D_matrix = 1 - K_matrix
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        X_2d = mds.fit_transform(D_matrix)
        tsne = TSNE(n_components=2, perplexity=20, metric='precomputed', init="random", random_state=42)
        X_2d = tsne.fit_transform(D_matrix)
        labels = [answers[subject] for subject in subjects_order]
        
                
        if PLOT:
            plt.figure(figsize=(6, 5))
            scatter = plt.scatter(
                X_2d[:, 0], X_2d[:, 1],
                c=labels,              
                cmap='tab10',          
                s=250,                 
                edgecolor='k'
            )
            for i, label in enumerate(labels):
                plt.text(X_2d[i,0], X_2d[i,1], str(label),
                        color='black', fontsize=9,
                        ha='center', va='center')
                
            plt.title("2D Layout of Items (t-SNE)")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.colorbar(scatter, label='Cluster')
            plt.show()



    def shortest_path_kernel_aois(img: str, PLOT: bool = False):
        dl = DataLoader("../eyelink_data")
        img_file = dl.get_image(img)
        aois = dl.get_image_aois(image=img)

        # if PLOT:
        PlotUtils.plot_aois(aois=aois, title="Original image and predefined AOIs", image=img_file)


        trials = dl.get_image_trials(img)
        trial_numbers = dl.get_image_trials(img, False)
        answer_dfs = dl.get_image_answers(img)
        answers = {subject: answer_dfs[subject].query(f"trial == {trial_numbers[subject]}")["response"].item() for subject in trials.keys()}
        if PLOT:
            PlotUtils.wrap_dict_plot(trials, n_wrap=3, plotting_function=PlotUtils.trace_plot, title="Subject", image=img_file)



        # MATCHING TO NEAREST AOI 

        trials_clustered = {subject: GraphUtils.cluster_to_aois(trial, aois) for (subject, trial) in trials.items()}

        if PLOT:
            PlotUtils.wrap_dict_plot(trials_clustered, n_wrap=3, plotting_function=PlotUtils.clustering_plot, title="Subject", image=img_file)



        # TRANSITION MATRCIES
        matrices = {subject: GraphUtils.transition_matrix(trial, aois=aois) for (subject, trial) in trials_clustered.items()}
        if PLOT:
            PlotUtils.wrap_dict_plot(matrices, n_wrap=3, plotting_function=PlotUtils.heatmap, title="Subject")



        # GRAPHS
        graphs = {subject: GraphUtils.transition_matrix_to_graph(matrix) for (subject, matrix) in matrices.items()}
        if PLOT:
            PlotUtils.wrap_dict_plot(graphs, n_wrap=3, plotting_function=PlotUtils.draw, title="Subject")




        # PRECOMPUTED KERNEL MATRIX
        subjects_order =list(matrices.keys())
        K_matrix = np.zeros(shape=(len(matrices), len(matrices)))
        for i, (subject1, matrix1) in enumerate(matrices.items()):
            for j, (subject2, matrix2) in enumerate(matrices.items()):
                K = ShortestPathKernel.get_kernel_value(matrix1, matrix2, kernel=ShortestPathKernel.inverse_abs_diff_kernel, normalize=True)
                K_matrix[i][j] = K

        if PLOT:
            PlotUtils.heatmap(K_matrix, title="Pairwise similarity")


        D_matrix = 1 - K_matrix
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        X_2d = mds.fit_transform(D_matrix)
        tsne = TSNE(n_components=2, perplexity=20, metric='precomputed', init="random", random_state=42)
        X_2d = tsne.fit_transform(D_matrix)
        labels = np.array([answers[subject] for subject in subjects_order])
        
        # if PLOT:        
        plt.figure(figsize=(6, 5))
        scatter = plt.scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=labels,              
            cmap='tab10',          
            s=250,                 
            edgecolor='k'
        )
        for i, label in enumerate(labels):
            plt.text(X_2d[i,0], X_2d[i,1], str(label),
                    color='black', fontsize=9,
                    ha='center', va='center')
            
        plt.title("2D Layout of Items (t-SNE)")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.colorbar(scatter, label='Cluster')
        plt.show()


        # LEAVE ONE OUT CROSS VALIDATION
        results = []
        alphas = np.arange(0, 1.01, 0.01)
        for alpha in alphas:
            loocv = LeaveOneOut()
            y_true, y_pred = [], []
            for train_index, test_index in loocv.split(K_matrix):
                K_train = K_matrix[np.ix_(train_index, train_index)]
                K_test = K_matrix[np.ix_(test_index, train_index)]
                model = KernelRidge(alpha=alpha, kernel="precomputed")
                model.fit(K_train, labels[train_index])
                            
                y_hat = model.predict(K_test)
                y_true.append(float(labels[test_index][0]))
                y_pred.append(float(y_hat[0]))

            # evaluate
            #for true, pred in zip(y_true, y_pred):
            #    print(f"{true} <> {pred}")

            mae = mean_absolute_error(y_true, y_pred)
            rmse = root_mean_squared_error(y_true, y_pred)

            # optional: round predictions back to [1,7]
            y_pred_rounded = np.clip(np.rint(y_pred), 1, 7)
            acc = np.mean(np.array(y_pred_rounded) == np.array(y_true))
            results.append(mae)
        
        fig, ax = plt.subplots(1,1)
        ax.set(title=img)
        ax.plot(alphas, results)


