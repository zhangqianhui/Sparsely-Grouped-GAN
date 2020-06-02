import tensorflow as tf
from IMLib.utils import *
import itertools

ATT_ID = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
          'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6,
          'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10,
          'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13,
          'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16,
          'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
          'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22,
          'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25,
          'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28,
          'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31,
          'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34,
          'Wearing_Hat': 35, 'Wearing_Lipstick': 36,
          'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}

class CelebA(object):

    def __init__(self, config):
        super(CelebA, self).__init__()

        self.data_dir = config.data_dir
        self.label_dir = config.label_dir
        self.dataset_name = 'CelebA'
        self.height, self.width= config.img_size, config.img_size
        self.channel = config.output_nc
        self.capacity = config.capacity
        self.batch_size = config.batch_size
        self.num_threads = config.num_threads
        self.chosen_att_names = config.chosen_att_names
        self.to_balance_att_names = config.to_balance_att_names

        self.img_names = np.genfromtxt(self.label_dir, dtype=str, usecols=0)
        self.img_paths = np.array([os.path.join(self.data_dir, img_name) for img_name in self.img_names[2:]])
        self.labels = self.read_txt(self.label_dir) #np.genfromtxt(self.label_dir, dtype=str, usecols=range(0, 41), delimiter='[/\s:]+')

        assert len(self.labels) == len(self.img_paths)

        self.train_images_list = self.img_paths[0:200599, ...]
        self.test_images_list = self.img_paths[200599:, ...]
        self.train_label = self.labels[0:200599, ...]
        self.test_label = self.labels[200599:, ...]

        pos = list(filter(lambda x:x==1, self.train_label[:, 31]))
        neg = list(filter(lambda x:x==0, self.train_label[:, 31]))

        print("pos", len(pos), "neg", len(neg))

        self.train_images_list, self.train_label = \
            self.balance(self.train_images_list, self.train_label, self.to_balance_att_names, balance_ratios=[0.55]*len(self.to_balance_att_names))

        pos = list(filter(lambda x:x==1, self.train_label[:, 31]))
        neg = list(filter(lambda x:x==0, self.train_label[:, 31]))

        print("pos", len(pos), "neg", len(neg))

        self.train_label = self.train_label[:, np.array([ATT_ID[att_name] for att_name in self.chosen_att_names])]
        self.test_label = self.test_label[:, np.array([ATT_ID[att_name] for att_name in self.chosen_att_names])]

    def read_images(self, input_queue):

        content = tf.read_file(input_queue)
        img = tf.image.decode_jpeg(content, channels=self.channel)
        img = tf.cast(img, tf.float32)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.crop_to_bounding_box(img, 20, 0, 178, 178)
        img = tf.image.resize_images(img, (self.height, self.width))

        return img / 127.5 - 1.0

    def read_txt(self, txt_path):

        p = open(txt_path, 'r')
        next(p)
        next(p)
        lines = p.readlines()
        labels = []
        for i, line in enumerate(lines):
            line = line.replace('\n', '')
            list = line.split()
            label = [(int(item) + 1)/2 for item in list[1:]]
            labels.append(label)

        return np.array(labels)

    def input(self):

        train_images = tf.convert_to_tensor(self.train_images_list, dtype=tf.string)
        train_label = tf.convert_to_tensor(self.train_label, dtype=tf.int32)
        train_queue = tf.train.slice_input_producer([train_images, train_label], shuffle=True)
        train_label_queue = train_queue[1]
        train_images_queue = self.read_images(input_queue=train_queue[0])

        test_images = tf.convert_to_tensor(self.test_images_list, dtype=tf.string)
        test_label = tf.convert_to_tensor(self.test_label, dtype=tf.int32)
        test_queue = tf.train.slice_input_producer([test_images, test_label], shuffle=False)
        test_label_queue = test_queue[1]
        test_images_queue = self.read_images(input_queue=test_queue[0])

        batch_image1, batch_label1 = tf.train.shuffle_batch([train_images_queue, train_label_queue],
                                                batch_size=self.batch_size,
                                                capacity=self.capacity,
                                                num_threads=self.num_threads,
                                                min_after_dequeue=10)

        batch_image2, batch_label2 = tf.train.batch([test_images_queue, test_label_queue],
                                                batch_size=self.batch_size,
                                                capacity=50,
                                                num_threads=1)

        return batch_image1, batch_label1, batch_image2, batch_label2

    def balance(self, img_paths, labels, to_balance_att_names, balance_ratios):

        assert len(to_balance_att_names) == len(balance_ratios)
        if to_balance_att_names == []:
            return img_paths, labels
        print(balance_ratios)
        to_balance_att_name = to_balance_att_names[0]
        balance_ratio = balance_ratios[0]
        labels_to_balance = labels[:, ATT_ID[to_balance_att_name]]
        idx_0 = np.argwhere(labels_to_balance == 0).squeeze()
        idx_1 = np.argwhere(labels_to_balance == 1).squeeze()

        if balance_ratio == 'only_neg':
            img_paths = img_paths[idx_0]
            labels = labels[idx_0]
            img_paths, labels = self.balance(img_paths, labels, to_balance_att_names[1:], balance_ratios[1:])

        elif balance_ratio == 'only_pos':
            img_paths = img_paths[idx_1]
            labels = labels[idx_1]
            img_paths, labels = self.balance(img_paths, labels, to_balance_att_names[1:], balance_ratios[1:])
        else:
            if len(idx_0) < len(idx_1) and len(idx_0) / len(idx_1) < balance_ratio and len(idx_0) != 0:
                idx_0, idx_1 = zip(*zip(itertools.cycle(idx_0), idx_1))
                idx_0, idx_1 = np.random.permutation(idx_0), np.array(idx_1)
                idx_0 = idx_0[:int(np.ceil(len(idx_1) * balance_ratio))]
            elif len(idx_1) < len(idx_0) and len(idx_1) / len(idx_0) < balance_ratio and len(idx_1) != 0:
                idx_0, idx_1 = zip(*zip(idx_0, itertools.cycle(idx_1)))
                idx_0, idx_1 = np.array(idx_0), np.random.permutation(idx_1)
                idx_1 = idx_1[:int(np.ceil(len(idx_0) * balance_ratio))]
            img_paths_0 = img_paths[idx_0]
            labels_0 = labels[idx_0]
            img_paths_1 = img_paths[idx_1]
            labels_1 = labels[idx_1]

            img_paths_0, labels_0 = self.balance(img_paths_0, labels_0, to_balance_att_names[1:], balance_ratios[1:])
            img_paths_1, labels_1 = self.balance(img_paths_1, labels_1, to_balance_att_names[1:], balance_ratios[1:])
            img_paths = np.concatenate((img_paths_0, img_paths_1))
            labels = np.concatenate((labels_0, labels_1))

        return img_paths, labels
