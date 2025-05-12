import h5py

def print_hdf5_structure(file_path):
    with h5py.File(file_path, 'r') as f:
        print("模型文件结构：")
        # 递归打印所有键和子组
        def print_group(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"组: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"数据集: {name} (形状: {obj.shape})")
        f.visititems(print_group)

# 替换为你的模型路径
print_hdf5_structure('training_history/2019-12-19 01%3A53%3A15.480800.hdf5')