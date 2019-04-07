import os,shutil

src_dir = "./dataset/train/"
des_dir = "./dataset/val/"
def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件

f = open(r"./dataset/test_list.txt")
for l in f.readlines():
    src_file = src_dir + l.split()[1]
    des_file = des_dir + l.split()[1]
    if not os.path.isfile(src_file):
        print("%s not exist!"%(l))
    else:
        shutil.move(src_file,des_file)


