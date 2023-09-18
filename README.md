# Master Thesis
Title: Automated labeling of plain-text privacy policies via machine learning by analysis of label strings for DPV mapping

# Libraries Used
Scikit-learn, Matplotlib, NLTK, Gensim, Skmultilearn, Tensorflow, and Oputna 

# Results 

ML Model Results
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
    <td>0.71</td>
    <td>0.40</td>
    <td>0.51</td>
    <td>0.60</td>
    <td>0.46</td>
    <td>0.52</td>
    <td>0.61</td>
    <td>0.47</td>
    <td>0.53</td>
  </tr>
  <tr>
    <td>RF</td>
    <td>0.94</td>
    <td>0.20</td>
    <td>0.33</td>
    <td>0.91</td>
    <td>0.16</td>
    <td>0.28</td>
    <td>0.62</td>
    <td>0.48</td>
    <td>0.52</td>
  </tr>
  <tr>
    <td>K-NN</td>
    <td>0.59</td>
    <td>0.41</td>
    <td>0.48</td>
    <td>0.48</td>
    <td>0.47</td>
    <td>0.48</td>
    <td>0.51</td>
    <td>0.54</td>
    <td>0.50</td>
  </tr>
</tbody>
</table>

DL Model Results

| DL Model | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| CNN      |   0.58    |  0.63  |   0.60   |
| LSTM     |   0.54    |  0.56  |   0.55   |
| Hybrid   |   0.56    |  0.66  |   0.62   |
