# Master Thesis
Title: Automated labeling of plain-text privacy policies via machine learning by analysis of label strings for DPV mapping

# Libraries Used
Scikit-learn, Matplotlib, NLTK, Gensim, Skmultilearn, Tensorflow, and Oputna 

# Results 

ML Model Results

|       |        Binary Relevance       |        Classifier Chain       |         Label Powerset        |
|-------|-------------------------------|-------------------------------|-------------------------------|
|       | Precision | Recall | F1-Score | Precision | Recall | F1-Score | Precision | Recall | F1-Score |
|-------|-------------------------------|-------------------------------|-------------------------------|
|  SVM  |    0.73   |  0.46  |   0.56   |    0.66   |  0.43  |   0.52   |    0.53   |  0.47  |   0.50   |
|  LR   |    0.71   |  0.40  |   0.51   |    0.60   |  0.46  |   0.52   |    0.61   |  0.47  |   0.53   |
|  RF   |    0.94   |  0.20  |   0.33   |    0.91   |  0.16  |   0.28   |    0.62   |  0.48  |   0.52   |
|  K-NN |    0.59   |  0.41  |   0.48   |    0.48   |  0.47  |   0.48   |    0.51   |  0.54  |   0.50   |

DL Model Results

| DL Model | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| CNN      |   0.58    |  0.63  |   0.60   |
| LSTM     |   0.54    |  0.56  |   0.55   |
| Hybrid   |   0.56    |  0.66  |   0.62   |
