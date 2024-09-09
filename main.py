import numpy as np
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs #only for create n dimentional dataset to test the code
import sys
import warnings

def loadDataset(datasetName):
    with open("../"+datasetName+".csv", mode='r') as infile:
        reader = csv.reader(infile)
        data = [row for row in reader]
    return data

def rangeQuery(data, pointIndex, e, p):
    neighbors = []
    point = data[pointIndex][:-1]
    for i, row in enumerate(data):
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
            if len(neighbors) >= minPts:
                data[pointIndex][-1] = "noise"
                label += 1
                data[pointIndex][-1] = str(label)
                s = [x for x in neighbors if x != pointIndex]
                while s:
                    i = s.pop(0)
                    if data[i][-1] == "noise":
                        data[i][-1] = str(label)
                    if data[i][-1] == "undefined":
                        data[i][-1] = str(label)
                        new_neighbors = rangeQuery(data, i, e, distanceFunction)
                        if len(new_neighbors) >= minPts:
                            s.extend([x for x in new_neighbors if x not in s])

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

def minkowskiDistance(pointOne, pointTwo, p):
    if len(pointOne) == len(pointTwo):
        tot = 0
        for i in range(len(pointOne)):
            tot = np.abs(pointOne[i] - pointTwo[i]) ** p
        return tot ** (1 / p)

def pipeline():
    datasets = ["blobs", "noisy_circles", "noisy_moons"]
    e = [0.1, 0.3, 0.5]
    minPts = [5, 20, 30]
    distanceTypes = ["euclidean", "manhattan"]
    p = [1, 2]

    for dataset in datasets:
        for e in e:
            for minPts in minPts:
                for distance_type in distanceTypes:
                        for p in p:
                            print(
                                f"Testing dataset: {dataset}, e={e}, minPts={minPts}, distanceType={distance_type} (p={p})")
                            data = loadDataset(dataset)
                            data_float = []
                            for row in data:
                                row_float = [float(value) for value in row]
                                data_float.append(row_float)
                                data_float[-1].append("undefined")
                            dbscan(data_float, e, minPts, p)
                            plotClusters(data_float, dataset, e, minPts, f"{distance_type}_p{p}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        pipeline()
    if len(sys.argv) > 6:
        warnings.warn("You have insert too many input arguments", UserWarning)
        raise ValueError("Execution stopped due to have pass too many arguments.")
    datasetName = "blobs"
    if len(sys.argv) > 1:
        if sys.argv[1] == "blobs":
            datasetName = "blobs"
        elif sys.argv[1] == "noisy_circles":
            datasetName = "noisy_circles"
        else:
            datasetName = "noisy_moons"

    data = loadDataset(datasetName)

    #Try this!
    #centers = 2
    #data, labels = make_blobs(n_samples=1600, centers=centers, n_features=3, random_state=42)

    data_float = []
    for row in data:
        row_float = [float(value) for value in row]
        data_float.append(row_float)
        data_float[-1].append("undefined")
    p=2
    distanceType = "euclidean"
    if len(sys.argv) <= 3:
        e = 0.2
        minPts = 20
    elif len(sys.argv) == 4:
        e = float(sys.argv[3])
        minPts = 20
    elif len(sys.argv) == 5:
        e = float(sys.argv[3])
        minPts = float(sys.argv[4])
    elif len(sys.argv) == 6:
        e = float(sys.argv[3])
        minPts = float(sys.argv[4])
        distanceType = sys.argv[5]
        if sys.argv[5] == distanceType:
            p = 2
        elif sys.argv[5] == distanceType:
            p = 1
        elif sys.argv[5] == distanceType:
            p = int(sys.argv[5])

    dbscan(data_float, e, minPts, p)

    plotClusters(data_float, datasetName, e, minPts, distanceType)
