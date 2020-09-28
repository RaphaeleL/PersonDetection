import glob, os, argparse

def index():
    train_data = opt.train
    valid_data = opt.valid
    pic_formats = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.tif', '.webp']
    num_train = 0
    num_valid = 0
    num_all = 0
    open(train_data, "w")
    open(valid_data, "w")

    for pic_format in pic_formats:
        #Train
        with open(train_data, "a") as axlspd:
            for f in glob.glob(r'../data/images/train/**/*'+pic_format, recursive=True):
                fpath, fname = os.path.split(f)
                rname, extname = os.path.splitext(fname)
                dirtup = (f, rname)
                axlspd.write(f+'\n')
                num_train+=1

        #Valid
        with open(valid_data, "a") as axlspd:
            for f in glob.glob(r'../data/images/valid/**/*'+pic_format, recursive=True):
                fpath, fname = os.path.split(f)
                rname, extname = os.path.splitext(fname)
                dirtup = (f, rname)
                axlspd.write(f+'\n')
                num_valid+=1
        
    print("Finished! Found: ", num_train, " train pictures, ", num_valid, " valid pictures, ", num_all, " pictures")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='../data/train.txt', help='*.txt file to index /data/images/train')
    parser.add_argument('--valid', type=str, default='../data/valid.txt', help='*.txt file to index /data/images/valid')
    opt = parser.parse_args()
    print(opt)
    index()
