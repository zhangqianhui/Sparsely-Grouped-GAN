from Dataset import CelebA
from SGGAN import SGGAN
import os
from config.test_options import TestOptions

opt = TestOptions().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

if __name__ == "__main__":
    m_ob = CelebA(opt)
    sggan = SGGAN(m_ob, opt=opt)
    sggan.build_test_model()
    sggan.test()

