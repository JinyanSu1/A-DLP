
### Installation

```bash
git clone https://github.com/JinyanSu1/A-DLP.git
cd A-DLP
pip install -e verl
pip install packaging
pip install ninja
pip install flash-attn --no-build-isolation
pip install -e .
```


### Train Models
```bash 
ray start --head --port=6381 --dashboard-port=8266
bash scripts/train.sh
```


Make sure to change the ```data.train_files``` and ```data.val_files``` to the absolute path of the data.

### Evaluate Models
```bash
bash scripts/eval.sh
```
Make sure to change the ```MODEL_PATH``` to your model path and ```data.path``` to the absolute path of the testing data.




## Acknowledgments

This codebase is built on top of [l1](https://github.com/cmu-l3/l1)




