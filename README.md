## 数据集基本情况
    
* 文件基本说明
    * 共有 9947个训练样本, 2365个测试样本
    * 第0列(id)       ：文章id  
    * 第1列(article)  ：「字」级别上的表示 
    * 第2列(word_seg) ：「词」级别上的表示  
    
       
* 类别统计
    * 主题统计（共 10 个主题）
      * 动力: 2732, 价格: 1273, 油耗: 1082, 操控: 1036, 舒适性: 931
      * 配置: 853, 安全性: 573, 内饰: 536, 外观: 489, 空间: 442
    
    * 情感统计（情感分 3 类）
      * -1: 1616, 0: 6661, 1: 1670

## processed_data/数据文件说明
    
* saved_weight/  各种格式保存的权值文件

* vector/ 保存数据向量化的结果
		
* result/ 保存结果相关文件


## utils 下部分工具方法说明

* `json_util`, 将`dict`类型保存到json文件可以看到数据而且读入方便，个人觉得比pickle更适合保存dict和list类型
```python
from utils.path_util import from_project_root
import utils.json_util as ju
bdc_json_url = from_project_root("processed_data/phrase_level_bdc.json")
bdc_dict = ju.load(bdc_json_url)  # load from json file
ju.dump(bdc_dict, bdc_json_url)  # dump dict object to json file
```

* `data_util`, 提供一些读入和操作数据的公用方法
  * `train_dev_split`, 将csv数据文件8:2划分为固定的训练集和验证集

        
## Git使用说明
* 登录 Github 添加自己公钥到 Github 账号
  * `git-bash` 运行 `cat ~/.ssh/id_rsa.pub`，复制结果在[此处](https://github.com/settings/keys)选择`New SSH key`粘贴到`Key`输入框，提示文件不存在则先运行`ssh-keygen`一直点回车就行
  
* clone 项目
  * 在保存项目的位置 `git-bash` 运行 `git clone git@github.com:csJd/dg_text.git`，生成的 `dg_text` 为项目文件夹
  
* push 更改
  * 将代码复制到 `dg_text` 文件夹恰当位置，运行以下命令push更改，以后直接在dg_text文件夹下开发，后续修改了项目文件也是这样push更改：
```sh
git add .
git commit -m "message"
git pull
git push
```
  * 对git不熟悉可以使用Pycharm快捷键`Ctrl + T` pull， 然后`Ctrl + K` push

* pull 更改
  * pull 其他人对项目文件作更改, 运行 `git pull` 即可
  
* 使用git尽量不要修改别人写的代码，而是调用其中的方法
   

## Pycharm 建议设置

* 换行符统一为'\n'
  * File | Settings | Editor | Code Style
    * `Line separator` 选择 `Unix and OS X (\n)`

* 编码统一为`UTF-8`
  * File | Settings | Editor | File Encodings
    * Global 和 Project 都选择 `UTF-8`
    * 下方 Default encoding for properties files也选择 `UTF-8`

* Python Doc 使用 Google 风格
  * File | Settings | Tools | Python Integrated Tools
    * `Docstring format` 选择 `Google`

* 设置Python 默认代码
  * File | Settings | Editor | File and Code Templates
    * 选择 `Python Script` 粘贴以下内容, `${USER}`可换为自己想显示的昵称
    * 可以自己按需修改

```python
# coding: utf-8
# created by ${USER} on ${DATE}


def main():
    pass


if __name__ == '__main__':
    main()

```

* Python代码中的文件路径
  * 建议所有路径都使用 `utils` 包下的 `path_util.from_project_root` 方法得到绝对路径
  * 例如要访问`data/train_set.csv`时，先鼠标右键复制`train_set.csv`的相对路径，然后直接调用方法就好
```python
from utils.path_util import from_project_root
train_data_url = from_project_root('data/train_set.csv')
print(train_data_url)
```

