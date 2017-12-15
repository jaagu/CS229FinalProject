import os

def main():
   
    lst = sorted(os.listdir("TWOLAYERKERAS_models")[1:], key=lambda x:int(x.split('.')[1]))
    for filename in lst:
        if filename.endswith(".hdf5"):
            print(filename.split('.')[1])
    
    return

if __name__ == '__main__':
    main()
