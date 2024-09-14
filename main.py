from sklearn.datasets import make_blobs  # only for creating n-dimensional dataset to test the code
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import csv
from tqdm import tqdm

def rangeQuery(data, pointIndex, e, p):
    neighbors = []
    point = data[pointIndex][:-1]
    for i, row in enumerate(data):
        if i != pointIndex:
            neighborPoint = row[:-1]
            dTmp = minkowskiDistance(point, neighborPoint, p)
            if dTmp <= e:
                neighbors.append(i)
    return neighbors

def dbscan(data, e, minPts, distanceFunction):
    label = 0
    for pointIndex in tqdm(range(len(data)), desc="DBSCAN Progress"):
        if data[pointIndex][-1] == "undefined":
            neighbors = rangeQuery(data, pointIndex, e, distanceFunction)
            if len(neighbors) < minPts:
                data[pointIndex][-1] = "noise"
            else:
                label += 1
                data[pointIndex][-1] = str(label)
                while neighbors:
                    i = neighbors.pop(0)
                    if data[i][-1] == "noise":
                        data[i][-1] = str(label)
                    if data[i][-1] == "undefined":
                        data[i][-1] = str(label)
                        new_neighbors = rangeQuery(data, i, e, distanceFunction)
                        if len(new_neighbors) >= minPts:
                            neighbors.extend([x for x in new_neighbors if x not in neighbors])

def readFromFile(fileName):
    with open(fileName, mode='r') as infile:
        reader = csv.reader(infile)
        data = [row for row in reader]
    return data

import pdb

def minkowskiDistance(pointOne, pointTwo, p):
    if len(pointOne) == len(pointTwo):
        tot = 0
        for i in range(len(pointOne)):
            tot += np.abs(pointOne[i] - pointTwo[i]) ** p
        return tot ** (1 / p)

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
            if label == "noise":
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
            if label == "noise":
                plt.scatter(label_points[:, 0], label_points[:, 1], color='black', label='Noise', marker='x')
            else:
                plt.scatter(label_points[:, 0], label_points[:, 1], color=color, label=f'Cluster {label}')

        plt.legend()
        plt.title("databaseName="+datasetName+", e="+str(e)+", minPts="+str(minPts)+", distance type="+distanceType)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
def pipeline():
    datasets = ["../blobs.csv", "../noisy_circles.csv", "../noisy_moons.csv"]
    e = [0.1, 0.3, 0.5]
    minPts = [5, 20, 30]
    p = [1, 2]
    distanceType = ["euclidean", "manhattan"]
    clustersList = []
    for dataset in datasets:
        for i in e:
            for j in minPts:
                for k in range(0, len(p)):
                    data = readFromFile(dataset)
                    data_float = []
                    for row in data:
                        row_float = [float(value) for value in row]
                        data_float.append(row_float)
                        data_float[-1].append("undefined")
                    dbscan(data_float, i, j, p[k])
                    plotClusters(data_float, dataset, i, j, distanceType[k])
        clustersList.append(clustersElements(data_float))
    for cluster in clustersList:
        print(cluster[0])

def clustersElements(data):
    labels = {row[-1] for row in data}
    if 'noise' in labels:
        labels.remove('noise')
    clusterList = []
    for label in labels:
        cluster_points = [point for point in range(len(data)) if data[point][-1] == label]
        tmpString = ""
        comma = False
        tmpString += "("
        for point in cluster_points:
            if comma:
                tmpString += ","
            tmpString += str(point)
            comma = True
        tmpString += ")"
        clusterList.append(tmpString)
    clusterList.sort(reverse=True, key=lambda x: x[0])
    return clusterList


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'pipeline':
        pipeline()
    else:
        if len(sys.argv) > 6:
            warnings.warn("You have insert too many input arguments", UserWarning)
            raise ValueError("Execution stopped due to have pass too many arguments.")
        datasetName = "../blobs.csv"
        if len(sys.argv) > 1:
            if sys.argv[1] == "blobs":
                datasetName = "../blobs.csv"
            elif sys.argv[1] == "noisy_circles":
                datasetName = "../noisy_circles.csv"
            else:
                datasetName = "../noisy_moons.csv"
        data = readFromFile(datasetName)

        #Try this!
        #data, labels = make_blobs(n_samples=1600, centers=2, n_features=3, random_state=42)

        data_float = []
        for row in data:
            row_float = [float(value) for value in row]
            data_float.append(row_float)
            data_float[-1].append("undefined")
        p=2
        distanceType = "euclidean"
        if len(sys.argv) < 2:
            e = 0.2             #change this for 3D example, set e=1
            minPts = 20         #set minPts=5
        if len(sys.argv) == 3:
            e = float(sys.argv[2])
            minPts = 20
        elif len(sys.argv) == 4:
            e = float(sys.argv[2])
            minPts = float(sys.argv[3])
        elif len(sys.argv) == 5:
            e = float(sys.argv[2])
            minPts = float(sys.argv[3])
            p = sys.argv[4]
            if p == 1:
                distanceType = "euclidean"
            elif p == 2:
                distanceType = "manhattan"
            else:
                distanceType = "minowski"

        dbscan(data_float, e, minPts, p)
        plotClusters(data_float, datasetName, e, minPts, distanceType)
        clustersList = clustersElements(data_float)
        for cluster in clustersList:
            print(cluster)

