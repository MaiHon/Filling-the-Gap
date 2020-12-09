import os
import logging
from pathlib import Path
from datetime import datetime


log_name = datetime.now().strftime("%H:%M:%S__%Y-%m-%d")


class config:
    def __init__(self):
        self.root = Path('.')
        self.postFix = "v7"
        
        # logging set
        logging_path = self.root / f'logs/loggers_{self.postFix}'
        if not os.path.exists(logging_path):
            os.makedirs(logging_path, exist_ok=True)
        self.log_dir = f'{logging_path}/{log_name}.log'
        logging.basicConfig(filename=self.log_dir,
                        filemode='w',
                        level=logging.INFO, 
                        format='%(asctime)s  |  %(message)s') 


        # dataset
        self.annot = self.root / 'data' / 'annotations' / 'mpii_human_pose_v1_u12_1.mat'
        self.image = self.root / 'data' / 'images'
        
        self.flip_pairs = [
            (0, 5), (1, 4), (2, 3),
            (10, 15), (11, 14), (12, 13)
        ]
        self.joints_num = 16
        self.rotate_factor = 30
        self.random_mask_num = 5

        self.flip = True
        self.rotate = True
        self.affine_transform = True


        # debug
        self.cudnn_deterministic = True
        self.cudnn_enable = True
        self.cudnn_benchmark = True


        self.debug = False
        self.debug_path = self.root / f"debug_{self.postFix}"
        if not os.path.exists(self.debug_path) and self.debug:
            os.makedirs(self.debug_path, exist_ok=True)
                
                
        # model
        self.input_dim = 32
        self.latent_dim = 20
        self.encode_hidden_dims = [64, 128]
        self.decode_hidden_dims = [64]
        self.l_layers = len(self.encode_hidden_dims)
        
        # beta train
        self.beta_train = False
        self.beta = 4
        self.gamma = 1000
        self.C_max = 25
        self.C_stop_iter = 1e5

        # train
        self.device = "cuda"
        self.log_interval = 100
        self.init_train = True
        
        self.optim = "Adam"
        self.lr = 1e-3

        self.start_epoch = 0
        self.epochs = 200
        self.train_batch = 256
        self.valid_batch = 256
        
        self.checkpoint = self.root / f"models/weights_{self.postFix}"
        if not os.path.exists(self.checkpoint):
            os.makedirs(self.checkpoint, exist_ok=True)
        
        self.load_weights = True
        if self.load_weights:
            self.weight_path = self.root / f'models/weights_{self.postFix}/042_.pth.tar'
            # self.weight_path = self.root / f'weights_v2/075_.pth.tar'

        # valid
        self.valid_interval = 1
        self.valid_visualize = True
        self.valid_visualize_num = 10
        self.valid_visualization = self.root / f"debug/valid_visualize_{self.postFix}"
        if not os.path.exists(self.valid_visualization) and self.valid_visualize:
            os.makedirs(self.valid_visualization, exist_ok=True)


        logging.info(f"Batch Size       --> {self.train_batch}")
        logging.info(f"Latent Dims      --> {self.latent_dim}")
        logging.info(f"Hidden Dims      --> {self.encode_hidden_dims}")

        logging.info(f"Learning Rate    --> {self.lr}")
        logging.info(f"Joints Num       --> {self.joints_num}")
        logging.info(f"Rotate Factor    --> {self.rotate_factor}")
        logging.info(f"Random Mask Num  --> {self.random_mask_num}\n")


cfg = config()