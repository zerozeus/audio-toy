# 基于CNN的音频识别

通过`Inception v3`和`MobileNet`对音频数据生成的时频图进行分类，来判断特定的音频类型。

### 依赖

建议使用`Anaconda`安装依赖，在mac上使用时，`librosa`和自带的`matplotlib`可能存在兼容性问题。

- `librosa`
- `tensorflow`
- `scipy`
- `matplotlib`
- `numpy`

### 训练

如果需要自己训练模型，需要准备好音频数据和对应的标注数据。

- 首先将音频数据放入` ./data/audio/ ` ，支持大部分常见音频格式，例如mp3、wav、flac等，对于m4a可能需要额外安装解码包。

- 将标注数据写入`./data/train.xml`，xml格式类似于：

	```xml
	<dir audio_dir="audio">
		<file audio_file="1.m4a">
			<section type="1" start="19076" end="19233"/>
			<section type="1" start="19076" end="19233"/>
		</file>
		<file audio_file="2.m4a">
			<section type="1" start="19076" end="19233"/>
		</file>
	</dir>
    ```

	其中`audio_dir`指向` ./data/audio/ `，`audio_file`代表文件名或者相对路径，`type`是此文件对应的类型，详细信息定义在`./data/config.xml`，`start`和`end`分别代表此音频片段的起始和结束时间，单位是毫秒；

- 修改`./data/config.xml`中的内容，格式类似于：

	```xml
	<type>
		<item name="type1">0</item>
		<item name="type2">1</item>
	</type>
    ```

	需要注意这里的type值需要跟`./data/train.xml`保持一致。

- 数据准备好以后，按需修改`./src/ml/train_config.xml`中的内容，大部分保持默认即可，需要注意的是最后一项，`architecture`表示识别模型，因为项目是基于Google的CNN迁移学习脚本开发的，支持两种：`Inception v3` 和 `MobileNet`，前者的准确率最高，但模型大速度慢，后者与之相反。默认选用`Inception v3`。

    `MobileNet`支持多种规格，合法字段如`mobilenet_0.25_128_quantized`，第一个数字代表模型参数的多少，选择范围 [`1.0`, `0.75`, `0.50`, `0.25`] ，第二个参数代表模型的输入图片的尺寸，选择范围 [`224`, `192`, `160`, `128`] ，两者的数字越小代表模型越小，速度越快，准确度越低。

- 训练参数和数据准备完成后，进入项目根目录，执行train.py脚本，在默认参数下，最终生成的模型位于`audio-toy/src/ml/model/output_graph.pb`

	```shell
	cd ./audio-toy
	python src/train.py
	```

- 训练过程中的log会放在 `audio-toy/src/ml/temp/retrain_logs`中，可以使用`tensorboard`看到可视化的结果：

	```shell
	tensorboard --logdir audio-toy/src/ml/temp/retrain_logs
	```

### 使用

假设模型已经训练完毕，现在需要预测某个音频文件中某种特征声音出现的时机。同样进入项目根目录，执行apply.py脚本。如果路径均为之前的默认路径，那么可以直接使用下面的命令：

```shell
cd ./audio-toy
python src/apply.py \
        --audio /path/to/your/audio_file \
        --start 1000 \
        --end 5000
```

其中`audio`代表待预测的音频文件，`start`和`end`分别代表预测的起始和结束时间，单位是毫秒。程序会在console中输出预测结果。

### 关于人声识别

项目之初是希望在10毫秒量级上识别出人声，但是标准的人声数据难以寻找，而且还需要手动进行标注，目前没有找到合适的人声数据集，因此需要花费大量时间跟进数据问题。

- 目前人声数据集少且不平衡，目前人声训练数据总时长不到4分钟，按照25/15(ms)切片后可以生成9600/16000张时频图，其中浊音(voice)占了绝大部分(90%左右)，导致轻音(unvoice)识别不准确；同时人声数据绝大部分源自一名女生，数据缺少多样性；

- 即使在数据量严重匮乏的情况下，训练出的模型还是能够比较精确的定位人声起始点，***数据待补充***；

- 时频图的生成是基于stft，对于人声可能存在更好的特征，参见[基于 TensorflowLite 在移动端实现人声识别](https://www.infoq.cn/article/speaker-dentification-based-on-tensorflowlite)。

### 特性和潜在问题

- 基于迁移学习，绝大部分参数不用重新训练，因此模型收敛速度快；而且这两类模型可以方便的转换成tf-lite，可以进一步减少模型尺寸，有利于移动端部署。

- 将音频识别转换成图像识别，利用cv上成熟的模型框架，提升识别准确度。

- 模型基于Google迁移学习的脚本开发，支持`Inception v3`和`MobileNet`，前者的准确率最高，但模型大速度慢，后者与之相反。现在看两者的accuracy相差不大，不过还需要进一步验证；

- 音频数据增强，目前训练没有对原始音频数据做进一步增强，为了提升模型的泛化性，类似的噪声增强等方法是有必要的；

- 移动端集成，移动端的集成需要有两部分工作，首先需要用c++重新实现stft和时频图，其次 MobileNet中最小的模型大概是970Kb，对于移动端而言还是过大，需要转换成tflite;

### TODO

1. 人声数据集
2. 音频数据增强
3. tflite模型转换+移动端集成
4. 新的人声特征

### 参考

- [基于 TensorflowLite 在移动端实现人声识别](https://www.infoq.cn/article/speaker-dentification-based-on-tensorflowlite)
- [UrbanSound](https://urbansounddataset.weebly.com/)
- [Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification](https://arxiv.org/pdf/1608.04363.pdf)
- [tensorflow-for-poets-2](https://github.com/googlecodelabs/tensorflow-for-poets-2)
- [human-voice-dataset](https://github.com/vocobox/human-voice-dataset)
