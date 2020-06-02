from Dataset import CelebA
from SGGAN import SGGAN
from SGGAN22 import SGGAN as SGGAN22
from SGGAN23 import SGGAN as SGGAN23
import os
from config.train_options import TrainOptions

opt = TrainOptions().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

if __name__ == "__main__":

    m_ob = CelebA(opt)
    if "5_22" in opt.exper_name:
        sggan = SGGAN22(m_ob, opt=opt)
    elif "5_23" in opt.exper_name:
        sggan = SGGAN23(m_ob, opt=opt)
    elif "5_24" in opt.exper_name:
        sggan = SGGAN23(m_ob, opt=opt)
    elif "5_25" in opt.exper_name:
        sggan = SGGAN23(m_ob, opt=opt)
    elif "5_26" in opt.exper_name:
        sggan = SGGAN23(m_ob, opt=opt)
    else:
        sggan = SGGAN(m_ob, opt=opt)

    sggan.build_model()
    sggan.train()

