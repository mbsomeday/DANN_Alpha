# DANN_Alpha

Pytorch实现的部分改编的Domain-Adversarial Training of Neural Networks (DANN) 网络。用于行人分类任务，训练数据为自己的数据集。

### 1.超参数搜索
`python scripts\hp_search.py --isTrain --hp_dir path/to/save`


### 2.模型训练

采用经过超参数搜索获取的最佳超参数组合。

`python scripts\dann.py --isTrain`



### 模型测试

`python scripts\dann.py --test_ds_list D1 --weight_dir weights\path --test_txt txt\path `














