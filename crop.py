# from osgeo import osr, gdal
import numpy as np
import os
from PIL import Image
import time
from skimage import io
import cv2

def get_file_names(data_dir, file_type=None):
    if file_type is None:
        file_type = ['tif', 'tiff']
    result_dir = []
    result_name = []
    for maindir, subdir, file_name_list in os.walk(data_dir):  # disdir 为灾害事件的文件夹
        for filename in file_name_list:
            apath = maindir + '//' + filename
            apath = apath.replace("\\", "/")
            ext = apath.split('.')[-1]
            if ext in file_type:
                result_dir.append(apath)
                result_name.append(filename)
            else:
                pass
    return result_dir, result_name


def get_same_img(img_dir, img_name):
    result = {}
    for idx, name in enumerate(img_name):
        temp_name = ''
        for idx2, item in enumerate(name.split('_')[:-4]):
            if idx2 == 0:
                temp_name = temp_name + item
            else:
                temp_name = temp_name + '_' + item

        if temp_name in result:
            result[temp_name].append(img_dir[idx])
        else:
            result[temp_name] = []
            result[temp_name].append(img_dir[idx])
    return result

# 影像投影
# postgis的安装将会影响gdal对tif影像的投影
def assign_spatial_reference_byfile(src_path, dst_path):
    src_ds = gdal.Open(src_path, gdal.GA_ReadOnly)
    sr = osr.SpatialReference()
    sr.ImportFromWkt(src_ds.GetProjectionRef())
    geoTransform = src_ds.GetGeoTransform()
    dst_ds = gdal.Open(dst_path, gdal.GA_Update)
    dst_ds.SetProjection(sr.ExportToWkt())
    dst_ds.SetGeoTransform(geoTransform)
    dst_ds = None
    src_ds = None


def cut(in_dir, out_dir, file_type=None, out_type='png', out_size=1024):
    if file_type is None:
        file_type = ['tif', 'tiff']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    data_dir_list, _ = get_file_names(in_dir, file_type)
    count = 0
    print('Cut begining for ', str(len(data_dir_list)), ' images.....')
    for each_dir in data_dir_list:
        time_start = time.time()
        # image = gdal.Open(each_dir).ReadAsArray()

        # image = cv2.imread(each_dir,cv2.IMREAD_GRAYSCALE)[:,:,np.newaxis]
        image = cv2.imread(each_dir)[:,:,::-1]
        # image = io.imread(each_dir)[:,:,np.newaxis]

        # image = np.rot90(image, k=2, axes=(1, 2))

        cut_factor_row = int(np.ceil(image.shape[0] / out_size)) + 1  # 0
        cut_factor_clo = int(np.ceil(image.shape[1] / out_size)) + 1  # 1

        for i in range(cut_factor_row):
            for j in range(cut_factor_clo):

                if i == cut_factor_row - 1:
                    i = int(image.shape[0] / out_size - 1)
                else:
                    pass

                    if j == cut_factor_clo - 1:
                        j = int(image.shape[1] / out_size - 1)
                    else:
                        pass

                start_x = int(np.rint(i * out_size))
                start_y = int(np.rint(j * out_size))

                end_x = min(image.shape[0], int(np.rint((i + 1) * out_size)))
                end_y = min(image.shape[1], int(np.rint((j + 1) * out_size)))

                temp_image = image[start_x:end_x, start_y:end_y,: ]
                if temp_image.sum()==0:
                    continue
                # temp_image = image[start_x:end_x, start_y:end_y, :]

                if temp_image.shape[0] == 0 or temp_image.shape[1] == 0:
                    continue

                if temp_image.shape[0] < out_size or temp_image.shape[1] < out_size:
                    padx = int(out_size - temp_image.shape[0])
                    pady = int(out_size - temp_image.shape[1])
                    temp_image = np.pad(temp_image, ( (0, padx), (0, pady),(0, 0)), 'constant', constant_values=( (0, 65535), (0, 65535),(0, 0)))

                if temp_image[0][0][0] == 65535 and temp_image[0][out_size-1][0] == 65535 \
                    and temp_image[out_size-1][0][0] == 65535 and temp_image[out_size-1][out_size-1][0] == 65535:
                    continue

                print('temp_image:', temp_image.shape)
                out_dir_images = out_dir + '/' + each_dir.split('/')[-3] + '_' + each_dir.split('/')[-2] + '_' + each_dir.split('/')[-1].split('.')[0] \
                                 + '_x' + str(int(start_x / out_size)) + '_y' + str(int(start_y / out_size)) + '.' + out_type

                # out_image = temp_image.transpose(1, 2, 0)
                # out_image = Image.fromarray(np.uint16(temp_image[:,:,0]))
                out_image = Image.fromarray(temp_image)

                out_image.save(out_dir_images)

                # src_path = 'data/img/1.tif'  # 带地理坐标的影像
                # assign_spatial_reference_byfile(src_path, out_dir_images)

        count += 1
        print('End of ' + str(count) + '/' + str(len(data_dir_list)) + '...')
        time_end = time.time()
        print('Time cost: ', time_end - time_start)
    print('Cut Finsh!')
    return 0


def combine(data_dir, w, h, c, out_dir, out_type='tif', file_type=None):
    if file_type is None:
        file_type = ['tif', 'tiff']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    img_dir, img_name = get_file_names(data_dir, file_type)
    print('Combine begining for ', str(len(img_dir)), ' images.....')
    dir_dict = get_same_img(img_dir, img_name)
    count = 0
    for key in dir_dict.keys():
        temp_label = np.zeros(shape=(w, h, c), dtype=np.uint8)
        dir_list = dir_dict[key]
        for item in dir_list:
            name_split = item.split('_')
            x_start = int(name_split[-4])
            x_end = int(name_split[-3])
            y_start = int(name_split[-2])
            y_end = int(name_split[-1].split('.')[0])
            img = Image.open(item)
            img = np.array(img)
            temp_label[x_start:x_end, y_start:y_end, :] = img

        img_name = key + '.' + out_type
        new_out_dir = out_dir + '/' + img_name

        label = Image.fromarray(temp_label)
        label.save(new_out_dir)
        # src_path = 'data/img/1.tif'  # 带地理坐标影像
        # assign_spatial_reference_byfile(src_path, new_out_dir)
        count += 1
        print('End of ' + str(count) + '/' + str(len(dir_dict)) + '...')
    print('Combine Finsh!')

    return 0

def makelist(image_dir):
    f=open(os.path.join(image_dir,"tai_dataset.csv"),'w')
    f.write('name,type\n')
    for dirpath, dirnames, filenames in os.walk(os.path.join(image_dir,'post')):
        for fn in filenames:
            f.write(os.path.join(dirpath,fn)+",no-building\n")
    f.close()

if __name__ == '__main__':
    # gdal.PushErrorHandler('CPLQuietErrorHandler')
    ##### cut
    data_dir = 'E://work//disaster//data//20220624JiangnanFlood2//origin_png//3m'
    # 'E://work//disaster//data//20170814NepalFlood//landcover//png'
    # 'E://work//disaster//data//0220615PakistanFlood//dem//tif-clip'
    # 'E://work//disaster//data//0220615PakistanFlood//landcover//png-clip'
    # 'E://work//disaster//data//211001TailandFlood//landcover//png'
    # 'E://work//disaster//data//s20220624//jiangnan-planet0602'
    out_dir = 'E://work//disaster//data//20220624JiangnanFlood2//png_clip_3m'
    # 'E://work//disaster//data//20170814NepalFlood//landcover//png-clip'
    # 'E://work//disaster//data//0220615PakistanFlood//dem//tif-clip-224'
    # 'E://work//disaster//data//0220615PakistanFlood//landcover//png-clip-224'
    # 'E://work//disaster//data//211001TailandFlood//landcover//png-clip'
    # 'E://work//disaster//data//s20220624//jiangnan-planet-clip0602'

    file_type = ['png']
    out_type = 'png'
    # cut_size = 896
    cut_size = 224

    cut(data_dir, out_dir, file_type, out_type, cut_size)
    # makelist(out_dir)

    ##### combine
#    data_dir='F:/Level1/cut_960'
#    h=3072
#    w=1792
#    c=3
#    out_dir='F:/Level1'
#    out_type='tif'
#    file_type=['tif']
#
#    combine(data_dir, w, h, c, out_dir, out_type, file_type)