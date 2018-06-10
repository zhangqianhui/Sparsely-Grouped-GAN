import os
import errno
import numpy as np
import scipy
import scipy.misc

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_image(image_path , image_size, is_crop= True, resize_w= 64, is_grayscale= False, is_test=False):
    return transform(imread(image_path , is_grayscale), image_size, is_crop , resize_w, is_test=is_test)

def transform(image, npx = 64 , is_crop=False, resize_w=64, is_test=False):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image , npx , resize_w = resize_w, is_test=is_test)
    else:
        cropped_image = image
        cropped_image = scipy.misc.imresize(cropped_image ,
                            [resize_w , resize_w])
    return np.array(cropped_image)/127.5 - 1

def center_crop(x, crop_h , crop_w=None, resize_w=64, is_test=False):

    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))

    if not is_test:
        rate = np.random.uniform(0, 1, size=1)
        if rate < 0.5:
            x = np.fliplr(x)
    return scipy.misc.imresize(x[20:218 - 20, 0: 178], [resize_w, resize_w])

def save_images(images, size, image_path, is_ouput=False):
    return imsave(inverse_transform(images, is_ouput), size, image_path)

def imread(path, is_grayscale=False):

    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def merge(images, size):

    if size[0] + size[1] == 2:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w: i * w + w, :] = image

    else:

        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w: i * w + w, :] = image
    return img

def inverse_transform(image, is_ouput=False):

    if is_ouput == True:
        print image[0]
    result = ((image + 1) * 127.5).astype(np.uint8)
    if is_ouput == True:
        print result
    return result

def read_image_list(category):

    filenames = []
    print("list file")
    list = os.listdir(category)
    list.sort()
    for file in list:
        if 'jpg' or 'png' in file:
            filenames.append(category + "/" + file)
    print("list file ending!")

    length = len(filenames)
    perm = np.arange(length)
    np.random.shuffle(perm)
    filenames = np.array(filenames)
    filenames = filenames[perm]

    return filenames

def read_all_image_list(category):

    filenames = []
    print("list file")
    list = os.walk(category)
    for file in list:
        for img in file[2]:
            if 'jpg' in img:
                filenames.append(file[0] + "/" + img)
    print("list file ending!")

    length = len(filenames)
    perm = np.arange(length)
    np.random.shuffle(perm)
    filenames = np.array(filenames)
    filenames = filenames[perm]

    return filenames

class CelebA(object):

    def __init__(self, image_path, image_size, sp_type):

        self.dataname = "CelebA"
        self.dims = image_size * image_size
        self.channel = 3
        self.shape = [image_size, image_size, self.channel]
        self.image_size = image_size
        self.sp_type = sp_type
        self.train_data_list, self.train_lab_list1, self.train_lab_list2, self.train_lab_list3, self.train_lab_list4, \
        self.label_mask_1, self.label_mask_2, self.label_mask_hair  = self.load_celebA(image_path)
        self.test_data_list, self.test_lab_list1, self.test_lab_list2, self.test_lab_list3, self.test_lab_list4, self.test_mask_hair = self.load_test_celebA(image_path)
        self.test_data_list_forhair, self.test_lab_list_forhair = self.load_testforhair_celebA(image_path)

    def load_celebA(self, image_path):

        # get the list of image path
        images_list, images_label1, images_label2, images_label3, images_label4, label_mask_hair = read_image_list_file(image_path, is_test=False)

        label_mask_1 = np.zeros(shape=[len(images_list)])
        #for attribute hair color
        label_mask_2 = np.zeros(shape=[len(images_list)])

        flag = self.sp_type
        if flag == 0:

            label_data_num = len(images_list)
            label_data_num_for_hair = len(images_list)
            repeat = 0

        elif flag == 1:

            label_data_num = 10000
            label_data_num_for_hair = 30000
            repeat = 10

        else:

            label_data_num = 1000
            label_data_num_for_hair = 3000
            repeat = 50

        label_mask_1[0: label_data_num] = 1
        label_mask_2[0: label_data_num_for_hair] = 1

        #aug data (0) for all
        #aug data (10) for 10000
        #aug data (50) for 1000
        for i in range(repeat):

            images_list = np.concatenate((images_list, images_list[0:label_data_num_for_hair]))
            images_label1 = np.concatenate((images_label1, images_label1[0:label_data_num_for_hair]))
            images_label2 = np.concatenate((images_label2, images_label2[0:label_data_num_for_hair]))
            images_label3 = np.concatenate((images_label3, images_label3[0:label_data_num_for_hair]))
            images_label4 = np.concatenate((images_label4, images_label4[0:label_data_num_for_hair]))

            label_mask_1 = np.concatenate((label_mask_1, label_mask_1[0:label_data_num_for_hair]))
            label_mask_2 = np.concatenate((label_mask_2, label_mask_2[0:label_data_num_for_hair]))

            label_mask_hair = np.concatenate((label_mask_hair, label_mask_hair[0:label_data_num_for_hair]))

        images_onehot_label1 = np.zeros((len(images_label1), 2), dtype=np.float)
        for i, label in enumerate(images_label1):
            images_onehot_label1[i, int(images_label1[i])] = 1.0

        images_onehot_label2 = np.zeros((len(images_label2), 2), dtype=np.float)
        for i, label in enumerate(images_label2):
            images_onehot_label2[i, int(images_label2[i])] = 1.0

        images_onehot_label3 = np.zeros((len(images_label3), 2), dtype=np.float)
        for i, label in enumerate(images_label3):
            images_onehot_label3[i, int(images_label3[i])] = 1.0

        images_onehot_label4 = np.zeros((len(images_label4), 2), dtype=np.float)
        for i, label in enumerate(images_label4):
            images_onehot_label4[i, int(images_label4[i])] = 1.0

        return np.array(images_list), images_onehot_label1, images_onehot_label2, images_onehot_label3, images_onehot_label4 \
            , label_mask_1, label_mask_2, np.array(label_mask_hair)

    def load_test_celebA(self, image_path):

        # get the list of image path
        images_list, images_label1, images_label2, images_label3, images_label4, test_mask_hair = read_image_list_file(image_path, is_test=True)
        return np.array(images_list), np.array(images_label1), np.array(images_label2), np.array(images_label3), np.array(images_label4), np.array(test_mask_hair)

    def load_testforhair_celebA(self, image_path):
        # get the list of image path
        images_list, images_label = read_image_list_file2(image_path, is_test=True)
        return np.array(images_list), np.array(images_label)

    def getShapeForData(self, filenames, is_test=False):

        array = [get_image(batch_file, 128, is_crop=True, resize_w=128,
                           is_grayscale=False, is_test=is_test) for batch_file in filenames]
        sample_images = np.array(array)
        return sample_images

    def getNextBatch(self, batch_num=0, batch_size=64, is_shuffle=True):

        ro_num = len(self.train_data_list) / batch_size
        if batch_num % ro_num == 0 and is_shuffle:

            length = len(self.train_data_list)
            perm = np.arange(length)
            np.random.shuffle(perm)
            self.train_data_list = self.train_data_list[perm]
            self.train_lab_list1 = self.train_lab_list1[perm]
            self.train_lab_list2 = self.train_lab_list2[perm]
            self.train_lab_list3 = self.train_lab_list3[perm]
            self.train_lab_list4 = self.train_lab_list4[perm]

            self.label_mask_1 = self.label_mask_1[perm]
            self.label_mask_2 = self.label_mask_2[perm]
            self.label_mask_hair = self.label_mask_hair[perm]

        return self.train_data_list[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.train_lab_list1[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.train_lab_list2[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.train_lab_list3[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.train_lab_list4[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.label_mask_1[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.label_mask_2[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.label_mask_hair[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size]

    def getTestNextBatch(self, batch_num=0, batch_size=128, is_shuffle=True):

        ro_num = len(self.test_data_list) / batch_size
        if batch_num == 0 and is_shuffle:

            length = len(self.test_data_list)
            perm = np.arange(length)
            np.random.shuffle(perm)

            self.test_data_list = self.test_data_list[perm]

            self.test_lab_list1 = self.test_lab_list1[perm]

            self.test_lab_list2 = self.test_lab_list2[perm]

            self.test_lab_list3 = self.test_lab_list3[perm]

            self.test_lab_list4 = self.test_lab_list4[perm]

            self.test_mask_hair = self.test_mask_hair[perm]

        return self.test_data_list[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.test_lab_list1[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.test_lab_list2[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.test_lab_list3[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.test_lab_list4[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.test_mask_hair[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size]

    def getTest_forhairNextBatch(self, batch_num=0, batch_size=128, is_shuffle=True):

        ro_num = len(self.test_data_list_forhair) / batch_size
        if batch_num == 0 and is_shuffle:

            length = len(self.test_data_list_forhair)
            perm = np.arange(length)
            np.random.shuffle(perm)
            self.test_data_list_forhair = np.array(self.test_data_list_forhair)
            self.test_data_list_forhair = self.test_data_list_forhair[perm]
            self.test_lab_list_forhair = np.array(self.test_lab_list_forhair)
            self.test_lab_list_forhair = self.test_lab_list_forhair[perm]

            print "test shuffle"

        return self.test_data_list_forhair[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.test_lab_list_forhair[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size]

def read_image_list_file(category, is_test):

    end_num = 0
    if is_test == False:

        #number of test data
        start_num = 5001
        path = category + "celebA/"

    else:

        start_num = 4
        path = category + "celebA/"
        end_num = 5001

    list_image = []
    list_label1 = []
    list_label2 = []
    list_label3 = []
    list_label4 = []

    num_1 = 0
    num_2 = 0

    label_mask_hair = []
    lines = open(category + "list_attr_celeba.txt")
    li_num = 0

    for line in lines:

        if li_num < start_num:
            li_num += 1
            continue

        if li_num >= end_num and is_test == True:
            break

        file_name = line.split(' ', 1)[0]
        # black hairs
        flag_1 = line.split('1 ', 41)[8]
        # Blone hair
        flag_2 = line.split('1 ', 41)[9]
        #Brown hair
        flag_3 = line.split('1 ', 41)[11]

        if flag_1 == ' ' and flag_3 != ' ':

            #one-hot
            list_label3.append(0)
            label_mask_hair.append(1)

        elif flag_2 == ' ' and flag_3 != ' ':

            list_label3.append(1)
            label_mask_hair.append(1)

        else:

            list_label3.append(0)
            label_mask_hair.append(0)

        list_image.append(path + file_name)

        # for gender
        flag = line.split('1 ', 41)[20]

        if flag == ' ':
            list_label1.append(1)
        else:
            list_label1.append(0)

        # for smile
        flag = line.split('1 ', 41)[31]

        if flag == ' ':
            # one-hot
            list_label2.append(1)

        else:
            list_label2.append(0)

        #for lipstick
        flag = line.split('1 ', 41)[36]

        if flag == ' ':
            list_label4.append(1)
            num_1 += 1
        else:
            list_label4.append(0)
            num_2 += 1

        li_num += 1

    lines.close()

    return list_image, list_label1, list_label2 \
        , list_label3, list_label4, label_mask_hair

#Test data for attribute hair
def read_image_list_file2(category, is_test):

    end_num = 0
    if is_test == False:

        start_num = 5001
        path = category + "celebA/"

    else:

        start_num = 4
        path = category + "celebA/"
        end_num = 5001

    list_image_1 = []
    list_image_2 = []
    list_label1 = []
    list_label2 = []

    lines = open(category + "list_attr_celeba.txt")
    li_num = 0
    for line in lines:

        if li_num < start_num:
            li_num += 1
            continue

        if li_num >= end_num and is_test == True:
            break

        file_name = line.split(' ', 1)[0]
        # black hairs
        flag_1 = line.split('1 ', 41)[8]
        # Blone hair
        flag_2 = line.split('1 ', 41)[9]
        #Brown hair
        flag_3 = line.split('1 ', 41)[11]

        if flag_1 == ' ' and flag_3 != ' ':

            # one-hot
            list_image_1.append(path + file_name)
            list_label1.append(0)

        elif flag_2 == ' ' and flag_3 != ' ':

            list_image_2.append(path + file_name)
            list_label2.append(1)

        li_num += 1

    min_n = np.min([len(list_image_1), len(list_image_2)])
    lines.close()

    return list_image_1[0:min_n] + list_image_2[0:min_n]\
        , list_label1[0: min_n] + list_label2[0: min_n]