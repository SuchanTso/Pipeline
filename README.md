# Pipeline
deep learning method for pipeline

## Start up

### git 环境
参考 [git](https://blog.csdn.net/weixin_45811256/article/details/130925392)
### conda 环境
参考 [conda](https://zhuanlan.zhihu.com/p/1896552549621936802)

### python依赖
```shell
git clone git@github.com:SuchanTso/Pipeline.git
cd pipeline
conda create -n pipeline
conda activate pipeline
pip install -r requirements.txt
```
### train
since we have uploaded a simple pipeline in data directory, you can run the code in the following way:
```shell
sh script/train_GNN.sh
```
in which we've preset the parameters for training.