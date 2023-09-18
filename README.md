# Master Thesis
Title: Automated labeling of plain-text privacy policies via machine learning by analysis of label strings for DPV mapping

# Libraries Used
Scikit-learn, Matplotlib, NLTK, Gensim, Skmultilearn, Tensorflow, and Oputna 

# Results 

ML Model Results

|       |<td colspan=3>Binary Relevance |<td colspan=3>Classifier Chains|<td colspan=3>Label Powerset   |
|-------|-------------------------------|-------------------------------|-------------------------------|
|       | Precision | Recall | F1-Score | Precision | Recall | F1-Score | Precision | Recall | F1-Score |
|-------|-------------------------------|-------------------------------|-------------------------------|
|  SVM  |    0.73   |  0.46  |   0.56   |    0.66   |  0.43  |   0.52   |    0.53   |  0.47  |   0.50   |
|  LR   |    0.71   |  0.40  |   0.51   |    0.60   |  0.46  |   0.52   |    0.61   |  0.47  |   0.53   |
|  RF   |    0.94   |  0.20  |   0.33   |    0.91   |  0.16  |   0.28   |    0.62   |  0.48  |   0.52   |
|  K-NN |    0.59   |  0.41  |   0.48   |    0.48   |  0.47  |   0.48   |    0.51   |  0.54  |   0.50   |


<table>
<thead>
  <tr>
    <th></th>
    <th colspan="3">Binary Relevance</th>
    <th colspan="3">Classifier Chains</th>
    <th colspan="3">Label Powerset</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td></td>
    <td>Precision</td>
    <td>Recall</td>
    <td>F1-Score</td>
    <td>Precision</td>
    <td>Recall</td>
    <td>F1-Score</td>
    <td>Precision</td>
    <td>Recall</td>
    <td>F1-Score</td>
  </tr>
  <tr>
    <td>SVM</td>
    <td>0.73</td>
    <td>0.46</td>
    <td>0.56</td>
    <td>0.66</td>
    <td>0.43</td>
    <td>0.52</td>
    <td>0.53</td>
    <td>0.47</td>
    <td>0.50</td>
  </tr>
  <tr>
    <td>LR</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</tbody>
</table>

DL Model Results

| DL Model | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| CNN      |   0.58    |  0.63  |   0.60   |
| LSTM     |   0.54    |  0.56  |   0.55   |
| Hybrid   |   0.56    |  0.66  |   0.62   |
