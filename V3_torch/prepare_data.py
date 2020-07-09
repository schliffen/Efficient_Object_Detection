#
#
#
import sys, os, glob



if __name__ == '__main__':
    data_dir = '../data/'
    save_dir = './data/'

    imgs_dir = data_dir + 'images/'
    lbl_dir = data_dir + 'annotations/'

    imgs_list = glob.glob(imgs_dir + '*.jpg')

    num_data = 0

    for img in imgs_list:
        img_name = img.split('/')[-1]
        lbl_name = img_name.split('.')[0] + '.txt'
        # reading the annotation
        # check if annotation is also exists:
        try:
            with open(lbl_dir + lbl_name, 'r') as f:
                lbl_lines = f.readlines()
        except:
            continue
        for line in lbl_lines:
            line = line.split(' ')
            # class_ = line[0]
            # bbox = [line[i].split('\n')[0] for i in range(1,5)]
            with open( save_dir + 'train_img_list.txt', 'a' ) as f:
                f.write( line[0] + '_' + line[1] +'-'+ line[2] + '-'  +  line[3] + '-'+ line[4].split('\n')[0] + '_'+ img_name  + '\n' )

            num_data +=1
            print("num data: ", num_data)


