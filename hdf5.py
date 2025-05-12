import h5py

# 打开HDF5文件
file = h5py.File('2019-12-19 01%3A53%3A15.480800.hdf5', 'r')

# 读取数据集
dataset = file['sensor_data']

# 打印数据类型和维度
print("数据集的数据类型：", dataset.dtype)
print("数据集的维度：", dataset.shape)

# 读取数据
data = dataset[()]
print("数据：", data)

# 关闭文件
file.close()
