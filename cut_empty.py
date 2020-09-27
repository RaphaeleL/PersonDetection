import glob, os, argparse

def delete_empty():
    num_delete = 0
        #Train
    for f in glob.glob(r'data/coco/labels/train/**/*.txt', recursive=True):
                fpath, fname = os.path.split(f)
                rname, extname = os.path.splitext(fname)
                dirtup = (f, rname)
                if os.stat(str(f)).st_size == 0:
                        os.remove(str(f))
                        os.remove((str(f).replace('data/coco/labels/','data/coco/images/')).replace('.txt','.jpg'))
              
                    
        
    print("Finished! Found: ", num_delete, " empty labels")


if __name__ == '__main__':
    print("Start delete pictures with empty labels")
    delete_empty()
