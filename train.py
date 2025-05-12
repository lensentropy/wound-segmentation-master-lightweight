# from tensorflow.keras.optimizers import Adam
# from keras.callbacks import EarlyStopping
# from keras.models import load_model
# from keras.utils import CustomObjectScope
# 修正后的导入语句 ✅
import os
import urllib.parse

import h5py
import lr
from datetime import datetime
from tensorflow import timestamp
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects  # 替代 CustomObjectScope
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope

# from keras.utils.generic_utils import CustomObjectScope

from models.unets import Unet2D
from models.deeplab import Deeplabv3, relu6, DepthwiseConv2D, BilinearUpsampling
from models.FCN import FCN_Vgg16_16s
from models.SegNet import SegNet

from utils.learning.metrics import dice_coef, precision, recall
from utils.learning.losses import dice_coef_loss
from utils.io.data import DataGen, save_results, save_history, load_data


# manually set cuda 10.0 path
#os.system('export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}')
#os.system('export PATH=/usr/local/cuda-10.0/bin:/usr/local/cuda-10.0/NsightCompute-1.0${PATH:+:${PATH}}')

# Varibales and data generator
input_dim_x=224
input_dim_y=224
n_filters = 32
dataset = 'Medetec_foot_ulcer_224'
data_gen = DataGen('./data/' + dataset + '/', split_ratio=0.2, x=input_dim_x, y=input_dim_y)

######### Get the deep learning models #########

######### Unet ##########
# unet2d = Unet2D(n_filters=n_filters, input_dim_x=None, input_dim_y=None, num_channels=3)
# model, model_name = unet2d.get_unet_model_yuanqing()

######### MobilenetV2 ##########
model = Deeplabv3(input_shape=(input_dim_x, input_dim_y, 3), classes=1)
model_name = 'MobilenetV2'
# with CustomObjectScope({'relu6': relu6,'DepthwiseConv2D': DepthwiseConv2D, 'BilinearUpsampling': BilinearUpsampling}):
#     model = load_model('training_history/2019-12-19 01%3A53%3A15.480800.hdf5'
#                        , custom_objects={'dice_coef': dice_coef, 'precision':precision, 'recall':recall})
# model.load_weights('training_history/2019-12-19 01%3A53%3A15.480800.hdf5')
# --------------------------------------------------------------------------------------------
model_path = './training_history/2019-12-19 01%3A53%3A15.480800.hdf5'
assert os.path.exists(model_path), "文件不存在，请检查路径！"

# 对于完整模型
model = load_model(model_path,
                  custom_objects={
                      'relu6': relu6,
                      'DepthwiseConv2D': DepthwiseConv2D,
                      'BilinearUpsampling': BilinearUpsampling,
                      'dice_coef': dice_coef
                  })

# 仅加载权重（需先构建相同结构模型）
model.load_weights(model_path)
# --------------------------------------------------------------------------------------

######### Vgg16 ##########
# model, model_name = FCN_Vgg16_16s(input_shape=(input_dim_x, input_dim_y, 3))

######### SegNet ##########
# segnet = SegNet(n_filters, input_dim_x, input_dim_y, num_channels=3)
# model, model_name = segnet.get_SegNet()

# plot_model(model, to_file=model_name+'.png')

# training
batch_size = 2
epochs = 200
learning_rate = 1e-4
loss = 'binary_crossentropy'

es = EarlyStopping(monitor='val_dice_coef', patience=200, mode='max', restore_best_weights=True)
#training_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs
#                             , validation_split=0.2, verbose=1, callbacks=[])

model.summary()
model.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics=[dice_coef, precision, recall])
training_history = model.fit_generator(data_gen.generate_data(batch_size=batch_size, train=True),
                                       steps_per_epoch=int(data_gen.get_num_data_points(train=True) / batch_size),
                                       callbacks=[es],
                                       validation_data=data_gen.generate_data(batch_size=batch_size, val=True),
                                       validation_steps=int(data_gen.get_num_data_points(val=True) / batch_size),
                                       epochs=epochs)
### save the model weight file and its training history
# save_history(model, model_name, training_history, dataset, n_filters, epochs, learning_rate, loss, color_space='RGB',
#              path='./training_history/')
# path='./training_history/'
# filename = f"{model_name}_{dataset}_{epochs}epochs_{lr}lr_{timestamp}"
# with open(os.path.join(path, f"{filename}_history.json"), 'w') as f:
#     json.dump(history_dict, f, indent=4)

# 1. 创建目录（如果不存在）
os.makedirs("./training_history", exist_ok=True)

# 2. 生成合法文件名
timestamp = '2025.5.9'
file_path = f"./training_history/{timestamp}.hdf5"

# 3. 写入文件（示例）
try:
    with h5py.File(file_path, "w") as f:
        f.create_dataset("data", data=[1, 2, 3])
    print(f"文件保存成功: {file_path}")
except Exception as e:
    print(f"错误: {e}")


print(f"✅ 模型和训练记录已保存到: {file_path}{timestamp}.*")