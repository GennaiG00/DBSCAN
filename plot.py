import numpy as np
import matplotlib.pyplot as plt
def plotClusters(data, datasetName, e, minPts, distanceType):
    points = np.array([row[:-1] for row in data])
    labels = [row[-1] for row in data]
    unique_labels = set(labels)
    colors = plt.colormaps['tab20'](np.linspace(0, 1, len(unique_labels)))

    if len(points[0]) == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        for label, color in zip(unique_labels, colors):
            label_points = points[np.array(labels) == label]
            if label == "undefined":
                ax.scatter(label_points[:, 0], label_points[:, 1], label_points[:, 2], color='black', label='Noise',
                           marker='x')
            else:
                ax.scatter(label_points[:, 0], label_points[:, 1], label_points[:, 2], color=color,
                           label=f'Cluster {label}')

        ax.legend()
        ax.set_title("databaseName="+datasetName+", e="+str(e)+", minPts="+str(minPts)+", distance type="+distanceType)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
    else:
        plt.figure(figsize=(8, 6))

        for label, color in zip(unique_labels, colors):
            label_points = points[np.array(labels) == label]
            if label == "undefined":
                plt.scatter(label_points[:, 0], label_points[:, 1], color='black', label='Noise', marker='x')
            else:
                plt.scatter(label_points[:, 0], label_points[:, 1], color=color, label=f'Cluster {label}')

        plt.legend()
        plt.title("databaseName="+datasetName+", e="+str(e)+", minPts="+str(minPts)+", distance type="+distanceType)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
