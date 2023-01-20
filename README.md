# align-and-translate
My replication code for the paper [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473).


## The data
Download and prepare the data by running
```console
$ cd data && bash prepare_data.sh && cd ..
```

Optionally delete the downloaded files (and keep only the extracted data) with
```console
$ rm data/*.tgz data/*.tar
```

After the operation, the `data` dir should look like this (you can check it
with `tree data`):
```
data
├── dev
│   ├── enhi_v1
│   │   ├── newsdev2014-ref.en.sgm
│   │   ...
│   ├── news-test2008-ref.cs.sgm
│   ├── news-test2008-ref.cz.sgm
│   ...
├── files.txt
├── prepare_data.sh
├── test
│   ├── newstest2014-csen-ref.cs.sgm
│   ...
└── train
    ├── commoncrawl
    │   ...
    ├── europarl-v7
    │   ...
    ├── gigafren
    │   ...
    ├── nc-v9
    │   ...
    └── un
        ...
```
