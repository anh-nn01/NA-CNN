# NA-CNN
Official implementation of NA-CNN (Neural Additive CNN) for Explainable AI in Climate Science, published in SIGSPATIAL 2023. <br>
* Paper: <a href="https://dl.acm.org/doi/10.1145/3589132.3628369"> <b> Towards Inherently Interpretable Deep Learning for Accelerating Scientific Discoveries in Climate Science </b> </a>  <br>
* DOI: https://doi.org/10.1145/3589132.3628369 <br>
* Authors: Anh N. Nhu and Yiqun Xie

<b> TLDR </b>: We proposed a simple yet effective CNN architecture called Neural Additive CNN (NA-CNN) that can inherently interpret and highlight important / most relevant regions of interest to the prediction tasks. The highlighted regions are particularly important in several scientific domains, including Climate Science and Geoscience, to pinpoint relevant areas to physical events of interest.

---
NOTES: 
1. The implementation in the Jupyter notebook is self-contained, including data processing, model training, and XAI evaluations for ease of use domain scientists, 
2. We also provided a separate `models.py` file to provide quick insights into our model's definition and implmentation.
3. The SST dataset is from <a href="https://www.psl.noaa.gov/repository/entry/show?entryid=f45cf25c-bde2-44bd-bf3d-c943d92c0dd8"> <b> NOAA PSL Climate Data Repository </b> </a>. Please download the dataset and put it in the `./data` folder.
4. Since there are many visualizations in the Notebook, it might take 10-20 seconds to display the notebook on GitHub.

---
Thank you for your interest in our work. If we find our work useful to our research, we would be very appreciate it if you can consider citing our paper:

```
@inproceedings{10.1145/3589132.3628369,
author = {Nhu, Anh N and Xie, Yiqun},
title = {Towards Inherently Interpretable Deep Learning for Accelerating Scientific Discoveries in Climate Science},
year = {2023},
isbn = {9798400701689},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3589132.3628369},
doi = {10.1145/3589132.3628369},
abstract = {While deep learning models have high representation power and promising performances, there is often a lack of evidence to interpret potential reasons behind the predictions, which is a major concern limiting their usability for scientific discovery. We propose a Neural Additive Convolutional Neural Network (NA-CNN) to enhance the interpretability of the model to facilitate scientific discoveries in climate science. To investigate the interpretation quality of NA-CNN, we perform experiments on the El Ni\~{n}o identification task where the ground truth for El Ni\~{n}o patterns is known and can be used for validation. Experiment results show that compared to Spatial Attention and state-of-the-art post-hoc explanation techniques, NA-CNN has higher interpretation precision, remarkably improved physical consistency, and reduced redundancy. These qualities provide an encouraging ground for domain scientists to focus their analysis on potentially relevant patterns and derive laws governing phenomena with unknown physical processes.},
booktitle = {Proceedings of the 31st ACM International Conference on Advances in Geographic Information Systems},
articleno = {8},
numpages = {2},
keywords = {scientific discovery, deep learning, explainable AI},
location = {Hamburg, Germany},
series = {SIGSPATIAL '23}
}
```
