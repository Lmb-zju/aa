import pickle

path = 'F:/Code_Test/Recommand_Project/TianChi_NewsRec/task1/temp_results/itemcf_i2i_sim.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径

f = open(path, 'rb')
data = pickle.load(f)
data_key = list(data.keys())

print(data[data_key[0]])
print(len(data))
f.close()