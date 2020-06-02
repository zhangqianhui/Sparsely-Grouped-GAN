from .options import BaseOptions

class TestOptions(BaseOptions):

    def initialize(self, parser):

        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--use_sp', action='store_true', help='use spetral normalization')
        parser.add_argument('--n_att', type=float, default=1, help='number of attribute')
        
        self.isTrain = False
        return parser

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain
        self.print_options(opt)
        self.opt = opt

        return self.opt