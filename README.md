# Master Thesis
Title: Automated labeling of plain-text privacy policies via machine learning by analysis of label strings for DPV mapping

# Abstract 
The increase of websites has grown exponentially throughout the years. This, in turn, has brought about various changes in the way people interact with each other over the Internet. Unfortunately, these online interactions leave a significant amount of personal digital records collected and processed by service providers.
Privacy policies are legal documents that disclose how service providers collect, store, and share user data. However, users rarely read them and struggle to understand them due to their complex technical jargon. In addition, the entities constructing the privacy policies are unambitious in making them comprehensible to the users. Therefore, the users are unmotivated to read the privacy policies due to the complicated structure.
To this challenge, text processing using machine learning (ML) has become a prominent method to distill meaningful information from complex privacy policies. For instance, the development of tools that automatically annotate privacy policy sentences. The annotations can help users efficiently understand the critical areas of privacy policies. However, although there has been success in classifying privacy policies using machine learning, other studies have shown that deep learning (DL) algorithms yield better results.
This thesis will implement privacy preference-ready plain text policies. The user can define the preferences of the policy content, and the model will check if the policy meets the user’s expectations. Therefore, the ML or DL approach is implemented to analyze the content of plain text policy. Different ML and DL approaches will be implemented and benchmarked to identify which techniques provide high performance in predicting the labels. The labeled plain text policy will be converted to a JSON representation that can be used to negotiate privacy preferences based on the latest DPV.
The dataset used for the study is the OPP-115 dataset, which contains 115 websites and their privacy policies from various sectors, such as arts, business, etc., that are presented in natural language with their annotations. The label strings are selected from the OPP-115 dataset instead of the label data practices as it will provide users with further information and insights on the privacy policies presented to them. The OPP-115 dataset was annotated before the enactment of GDPR. Therefore, the latest DPV vocabulary will be mapped to the dataset. The mapped DPV vocabulary would modernize the labels in the OPP-155 dataset.

# Dataset
OPP-115 Dataset
- pretty_print
- sanitized_policies

# Libraries Used
Scikit-learn, Matplotlib, NLTK, Gensim, Skmultilearn, Tensorflow, and Optuna 

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
