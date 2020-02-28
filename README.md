## Instructions

- Download Graph Embeddings implementations (Forked):
```
>> git clone https://github.com/yoavnavon/GraphEmbedding.git
```
- Download datasets to `data/` folder

## Datasets

- Dynamobi
- [Youtube](http://networkrepository.com/soc-youtube-growth.php)
- [HepPh](https://snap.stanford.edu/data/cit-HepPh.html)
- [Twitter](https://snap.stanford.edu/data/ego-Twitter.html)

## Run Jupyter

```
>> jupyter notebook --no-browser --port=8080
>> ssh -N -L 8080:localhost:8080 ynavon26@crcfe02.crc.nd.edu
```

## Code Organization

- Every Dataset (Dynamobi, Youtube, HepPh, Twitter) has it's own code file with
the same name. Most of the training methods are in `dynamobi.py`.

## Results and Notebooks Organization

- All the results are in the folder `results/`. The ones with a number (e.g. 25_xx_.csv)
denotes the day in February it was writen. The ones without a number where made before
keeping track of the dates.

- Latest performance visualizations are in the `vis2.ipynb`notebook.