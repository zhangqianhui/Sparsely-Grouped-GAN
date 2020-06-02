from Dataset import CelebA
from SGGAN23 import SGGAN as SGGAN23
import os
from config.train_options import TrainOptions

opt = TrainOptions().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

if __name__ == "__main__":

    m_ob = CelebA(opt)
    sggan = SGGAN23(m_ob, opt=opt)
    sggan.build_model()
    sggan.train()

