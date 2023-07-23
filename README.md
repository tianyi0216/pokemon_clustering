# pokemon_clustering

This is a project I developed in Artificial Intelligence course at UW-Madison. It is an implementation of the agglomerative clustering algorithm from scratch using python and NumPy. This program performs agglomerative clustering on pokemons using 6 features. The program outputs the clusters based on those features. The program uses complete linkage.

To run the program, simply run the following command in the terminal:

``` 
python3 cluster.py num_pokemon
```

where num_pokemon is the number of pokemons you want to cluster. The program will output the clusters in a matplotlib graph.

It is similar to the scipy's `scipy.cluster.hierachy.linkage(features, method = "complete")` function.
