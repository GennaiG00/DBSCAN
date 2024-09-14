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
- `p=1` : Manhattan distance
- `p=2` : Euclidean distance
- `p=3…n` : Minowski distance
  
If you would test only the code, there is a pipeline that run dbscan 54 times and all
time there is a different setup. If you would run the pipeline, you must write `pipeline` after main.py like this:
```
python3 main.py pipeline
```
- `dataset=[blobs, noisy_circles, noisy_moon]`
- `e=[0.1, 0.3, 0.5]`
- `minPts=[5, 20, 30]`
- `distanceType=[Euclidean, Manhattan]`

### 3D Example
There is a commented line that is only for testing that the algorithm work with more than two dimensions. To try this you must uncommented the line:
```
data, labels = make_blobs(n_samples=1600, centers=2, n_features=3, random_state=42)
```
