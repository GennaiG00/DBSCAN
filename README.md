### DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is an unsupervised clustering algorithm. It groups together closely packed points based on density and labels points that don't belong to any cluster as noise.

## Code Explanation

The code is implemented in a simple way. You can run it from the command line and modify parameters such as:

- `e`: Radius (Epsilon)
- `minPts`: Density Threshold
- `datasetName`: Dataset Name
- `p`: Type of Distance (Euclidean, Manhattan, Minkowski)
  
You can run the code from the terminal (if you are in the project folder)
without any parameters, like this:
```
python3 main.py
```
In this case, the code runs with the default parameters:
- `e=0.2`
- `minPts=20`
- `datasetName=blobs`
- `p=2` (Euclidean distance)
  
If you want to change the parameters, you can specify them after main.py,
like this:
```
python3 main.py <datasetName> <e> <minPts> <p>
```
For the distance I use the same method(minowskiDistance) that calculate the
Minowski distance because this measurement it’s a generalization and just by
changing the value of p I can use:
- p=1 Manhattan distance
- p=2 Euclidean distance
- p=3…n Minowski distance
If you would test only the code, there is a pipeline that run dbscan 54 times and all
time there is a different setup.
- dataset=[blobs, noisy_circles, noisy_moon]
- e=[0.1, 0.3, 0.5]
- minPts=[5, 20, 30]
- distanceType=[Euclidean, Manhattan]
main()
There are some calls to functions for execute the code and the sys for insert value
from terminal.
dbscan()
In this part of code there are not many differences with the pseudocode. The
mechanism of the algorithm is, in short, a way to find clusters in the dataset.
There are three different tags:
- undefined: the point has not yet been inspected.
- noise: the point is only noise must not be part of a cluster.
- 1…n: if it has a number, it is included in a cluster.
When the algorithm is stops, all points have a tag(belong to a cluster).
In the pseudocode that was provided to us, there is only the difference that the if
condition use in reverse, but I don’t like to use continue in this way, so I change it.
I have added a progress bar to follow the process time.
rangeQuery()
rangeQuery() is another method that is combined with dbscan() and is it use only to
find the neighbors of a specific point that I pass to it. To find all neighbors, I pass to
it some parameters like:
- data: all points in the dataset.
- pointIndex: the specific point that I will find his neighbors.
- e: radius.
- distanceFunction: type of method to calculate the distance.
minowskiDistance() & plotClusters()
These two methods have only task: to calculate the distance between points (first)
and to plot the clusters (second).
