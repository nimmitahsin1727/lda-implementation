# latent-dirichlet-allocation-gensim

## About
Latent dirichlet allocation model from GENSIM library. 

Here, my main goal is to use my previously created preprocessed dataset [20-news-dataset-pre-processing](https://github.com/nimmitahsin1727/20-news-dataset-pre-processing) for GENSIM's LDA model [gensim.models.ldamulticore](https://radimrehurek.com/gensim/models/ldamulticore.html). And see the topics.

##  Outcome:
I've used same dataset, but in two different ways.
Here, the main difference is that in the second scenario I've added an extra layer of dictionary `dictionary.filter_extremes`.

### First scenario - [lda-gensim.ipynb](/lda-gensim.ipynb)

`
data_words => dictionary => doc2bow => lda_model => topics
`

**Topics:**
```js
[
  (0,
  '0.009*"line" + 0.007*"write" + 0.006*"right" + 0.005*"use" + 0.004*"make" + '
  '0.004*"like" + 0.004*"say" + 0.004*"know" + 0.004*"point" + 0.004*"file"'),
 (1,
  '0.010*"line" + 0.008*"bike" + 0.007*"write" + 0.006*"like" + 0.005*"file" + '
  '0.004*"good" + 0.004*"use" + 0.004*"work" + 0.004*"know" + 0.004*"time"'),
 (2,
  '0.007*"file" + 0.007*"line" + 0.006*"image" + 0.006*"write" + 0.005*"use" + '
  '0.004*"know" + 0.004*"gun" + 0.004*"need" + 0.003*"just" + 0.003*"like"'),
 (3,
  '0.007*"line" + 0.006*"use" + 0.005*"image" + 0.005*"bike" + 0.005*"write" + '
  '0.004*"gun" + 0.004*"like" + 0.004*"file" + 0.004*"just" + 0.003*"make"'),
 (4,
  '0.008*"gun" + 0.007*"line" + 0.006*"use" + 0.006*"state" + 0.006*"write" + '
  '0.005*"right" + 0.005*"make" + 0.004*"like" + 0.004*"say" + 0.004*"good"'),
 (5,
  '0.008*"gun" + 0.007*"line" + 0.006*"write" + 0.005*"use" + 0.005*"good" + '
  '0.004*"point" + 0.004*"like" + 0.004*"make" + 0.003*"say" + 0.003*"state"'),
 (6,
  '0.008*"use" + 0.008*"line" + 0.007*"gun" + 0.006*"write" + 0.005*"just" + '
  '0.005*"like" + 0.005*"know" + 0.004*"make" + 0.004*"file" + 0.004*"image"'),
 (7,
  '0.012*"line" + 0.007*"write" + 0.006*"image" + 0.004*"use" + 0.004*"make" + '
  '0.004*"gun" + 0.004*"know" + 0.003*"just" + 0.003*"like" + 0.003*"file"'),
 (8,
  '0.009*"write" + 0.007*"line" + 0.006*"file" + 0.006*"use" + 0.005*"know" + '
  '0.005*"gun" + 0.004*"graphic" + 0.003*"state" + 0.003*"need" + '
  '0.003*"good"'),
 (9,
  '0.012*"line" + 0.007*"write" + 0.007*"gun" + 0.005*"use" + 0.005*"say" + '
  '0.005*"make" + 0.004*"file" + 0.004*"like" + 0.004*"thing" + 0.004*"know"')
]
  ```

### Second scenario - [lda-gensim-with-token-filter.ipynb](/lda-gensim-with-token-filter.ipynb)

`
data_words => dictionary => dictionary.filter_extremes => doc2bow => lda_model => topics
`

**Topics:**
```js
[
  (0,
  '0.019*"gun" + 0.015*"use" + 0.012*"image" + 0.009*"like" + 0.008*"make" + '
  '0.006*"file" + 0.006*"right" + 0.006*"just" + 0.006*"know" + '
  '0.005*"program"'),
 (1,
  '0.009*"say" + 0.008*"make" + 0.008*"fbi" + 0.007*"right" + 0.007*"gun" + '
  '0.006*"just" + 0.006*"time" + 0.006*"start" + 0.006*"like" + 0.005*"good"'),
 (2,
  '0.016*"bike" + 0.010*"like" + 0.008*"right" + 0.008*"time" + 0.008*"just" + '
  '0.008*"say" + 0.007*"make" + 0.006*"dod" + 0.006*"gun" + 0.006*"use"'),
 (3,
  '0.025*"file" + 0.008*"good" + 0.006*"use" + 0.006*"know" + 0.005*"make" + '
  '0.005*"image" + 0.005*"world" + 0.005*"need" + 0.005*"like" + '
  '0.005*"graphic"'),
 (4,
  '0.018*"image" + 0.009*"graphic" + 0.009*"file" + 0.009*"point" + '
  '0.008*"use" + 0.007*"package" + 0.007*"know" + 0.006*"software" + '
  '0.006*"data" + 0.006*"look"'),
 (5,
  '0.009*"jpeg" + 0.009*"bike" + 0.008*"dod" + 0.008*"behanna" + 0.008*"use" + '
  '0.007*"right" + 0.007*"know" + 0.006*"state" + 0.006*"image" + '
  '0.006*"file"'),
 (6,
  '0.010*"use" + 0.008*"good" + 0.008*"knife" + 0.008*"know" + 0.007*"card" + '
  '0.007*"just" + 0.007*"gun" + 0.006*"say" + 0.006*"like" + 0.005*"bike"'),
 (7,
  '0.010*"know" + 0.010*"make" + 0.009*"helmet" + 0.009*"say" + 0.009*"bike" + '
  '0.008*"just" + 0.008*"use" + 0.007*"like" + 0.006*"child" + 0.006*"want"'),
 (8,
  '0.015*"use" + 0.010*"know" + 0.009*"like" + 0.008*"dog" + 0.007*"polygon" + '
  '0.006*"apr" + 0.006*"bike" + 0.006*"just" + 0.006*"point" + 0.006*"look"'),
 (9,
  '0.017*"gun" + 0.008*"state" + 0.008*"make" + 0.008*"file" + 0.007*"use" + '
  '0.007*"say" + 0.006*"law" + 0.006*"right" + 0.006*"know" + 0.006*"just"')
]
```

## Packages

- pandas
- nltk
- gensim

## Folder structure
```bash
├── lda-gensim.ipynb
├── lda-gensim-with-token-filter.ipynb
├── training_df.csv
├── testing_df.csv
├── README.md
└── .gitignore
```
