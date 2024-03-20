import os
import sys
from tqdm import tqdm
import datetime
import torch
import pickle
import numpy as np
import pathlib
import logging
import logging.config
from torch import nn
import time
import util.datasets as datasets
from util.transforms import DEFAULT_TRANSFORMS
# from models import ResidualBlock
from torchvision.models.resnet import Bottleneck

# log = logging.getLogger("sampleLogger")
# log.debug("In " + os.uname()[1])

class Activations:
    def __init__(self, model, dataloader, device, batch_size):
        """
        Parameters
        ----------
        model: Network
            The network we want to calculate the connectivity for.
        dataloader: Data
            The data we are drawing from, usuaully the test set.
        device: string
            "cpu" or "cuda"
        batch_size: int
            Batch size
        """
        self.activation = {}
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.batch_size = batch_size
        self.layers_dim = None
        self.layers_idx = None
        self.act_keys = None
        self.hook_handles = None

    def hook_fn(self, m, i, o):
        """Assign the activations/mean of activations to a matrix
        Parameters
        ----------
        self: type
            description
        m: str
            Layer's name
        i: type
            description
        o: torch tensor
            the activation function output
        """

        # tmp = o.detach()
        # if len(tmp.shape) > 2:
        #     self.activation[m] = torch.mean(tmp, axis=(2, 3))
        #     # activation[m] = tmp
        # else:
            # self.activation[m] = tmp
        self.activation[m] = o.detach()

    def hook_layer_idx(self, item_key, hook_handles):
        for module_idx, module in enumerate(self.model.named_modules()):
            if isinstance(module[1], nn.Conv2d) or \
                         isinstance(module[1], nn.Linear):
                if (module_idx == item_key):
                    hook_handles.append(module[1].register_forward_hook(self.hook_fn))
                    self.layers_dim.append(module[1].weight.shape)

    def hook_all_layers(self, layers_dim, hook_handles):
        """ Hook a handle to all layers that are interesting to us, such as
        Linear or Conv2d.
        Parameters
        ----------
        net: Network
            The network we're looking at
        layers_dim: list
            List of layers that are registered for saving activations
        hook_handles: type
            description

        If it is a sequential, don't register a hook on it but recursively
        register hook on all it's module children
        """
        for module in self.model.named_modules():
            if isinstance(module[1], nn.Conv2d) or \
                         isinstance(module[1], nn.Linear):
                hook_handles.append(module[1].register_forward_hook(self.hook_fn))
                layers_dim.append(module[1].weight.shape)

        # Recursive version
        # for name, layer in net._modules.items():
        #     if (isinstance(layer, nn.Sequential) or
        #             isinstance(layer, ResidualBlock) or
        #             isinstance(layer, Bottleneck)):
        #         self.get_all_layers(layer, layers_dim, hook_handles)
        #     elif (isinstance(layer, nn.Conv2d) or
        #           (isinstance(layer, nn.Linear))):
        #         # it's a non sequential. Register a hook
        #         hook_handles.append(layer.register_forward_hook(self.hook_fn))
        #         layers_dim.append(layer.weight.shape)

    def set_layers_idx(self):
        """Find all convolutional and linear layers and add them to layers_ind.
        """
        layers_idx = []
        layers_dim = []
        for module_idx, module in enumerate(self.model.named_modules()):
            if isinstance(module[1], nn.Conv2d) or \
                         isinstance(module[1], nn.Linear):
                layers_idx.append(module_idx)
                layers_dim.append(module[1].weight.shape)

        self.layers_dim = layers_dim
        self.layers_idx = layers_idx

    def get_layers_idx(self):
        if self.layers_idx == None:
            self.set_layers_idx()
        return self.layers_idx

    def set_act_keys(self):
        layers_dim = []
        hook_handles = []

        self.hook_all_layers(layers_dim, hook_handles)

        with torch.no_grad():
            names, X,_, y = next(iter(self.dataloader))
            X, y = X.to(self.device), y.to(self.device)
            self.model(X)
            act_keys = list(self.activation.keys())

        self.act_keys = act_keys
        self.hook_handles = hook_handles

    def get_act_keys(self):
        if self.act_keys == None:
            self.set_act_keys()
        return self.act_keys

    def get_act_layer(self, layers_dim, hook_handles):
        """ Hook a handle to all layers that are interesting to us, such as
        Linear or Conv2d.
        Parameters
        ----------
        net: Network
            The network we're looking at
        layers_dim: list
            List of layers that are registered for saving activations
        hook_handles: type
            description

        If it is a sequential, don't register a hook on it but recursively
        register hook on all it's module children
        """
        for module in self.model.named_modules():
            if isinstance(module[1], nn.Conv2d) or \
                         isinstance(module[1], nn.Linear):
                hook_handles.append(module[1].register_forward_hook(self.hook_fn))
                self.layers_dim.append(module[1].weight.shape)

        # Recursive version
        # for name, layer in net._modules.items():
        #     if (isinstance(layer, nn.Sequential) or
        #             isinstance(layer, ResidualBlock) or
        #             isinstance(layer, Bottleneck)):
        #         self.get_all_layers(layer, layers_dim, hook_handles)
        #     elif (isinstance(layer, nn.Conv2d) or
        #           (isinstance(layer, nn.Linear))):
        #         # it's a non sequential. Register a hook
        #         hook_handles.append(layer.register_forward_hook(self.hook_fn))
        #         layers_dim.append(layer.weight.shape)



    def get_data_dict(sefl):
        dict_filename = 'clear-to-foggy-dict.pkl'
        if os.path.exists(dict_filename):
            return pickle.load(open(dict_filename, 'rb'))
        data_dict = {}
        # 1. read the file containing the list of synthetically foggy images
        clear_data_dir = "/home/soheil/data/data_vocfog/vocfog_train"
        with open(clear_data_dir, 'r') as f:
            txt = f.readlines()
            clear_file_path = [line.strip().split()[0] for line in txt]

        foggy_data_dir = "/home/soheil/data/data_vocfog/vocfog_train_list"
        with open(foggy_data_dir, 'r') as f:
            txt = f.readlines()
            foggy_file_path = [line.strip().split()[0] for line in txt]

        for clear_img_dir in tqdm(clear_file_path):
            img_name = clear_img_dir.split('/')[-1].split('.')[0]
            for foggy_image_path in foggy_file_path:
                # foggy_name = foggy_image_path.split('/')[-1].split('.')[0] # .split('_')[0]
                foggy_name = "_".join(foggy_image_path.split('/')[-1].split('.')[0].split('_')[0:-1])
                if  img_name == foggy_name:
                    if data_dict.get(img_name) is None:
                        data_dict[img_name] = [clear_img_dir]
                    else:
                        data_dict[img_name].append(foggy_image_path)

        for elem in data_dict:
            data_dict[elem].sort()
            # if len(data_dict[elem]) != 10:
                # print(elem)
        pickle.dump(data_dict, open(dict_filename, 'wb'))

        return data_dict

    def create_dataset_list(sefl):
        data_dict = self.get_data_dict()
        for elem in data_dict:
            pass

    def estimate_uncertainty(self):
        # 1. read the file containing the list of synthetically foggy images
        # TODO: copy the clear image files to the same directory as the foggy
        # ones and copy the annotation .txt file alongside
        valid_path = "/home/soheil/data/data_vocfog/vocfog_test_list"
        mini_batch_size = 10
        valid_dataloader = datasets._create_data_loader(valid_path,
                                                        mini_batch_size,
                                                        self.model.hyperparams['height'],
                                                        n_cpu=8,
                                                        transform=DEFAULT_TRANSFORMS,
                                                        shuffle=False)

        # 2. for a file with a basename, find the foggy corresponding images
        # 3. run the clear image through the network and save the activations
        # 4. run the foggyfied images through the network
        # 5. calculate the weighted summation of the norm of the difference
        # between activations
        # 6. save the value alongside with the image name

        self.model.eval()
        ds_size = len(valid_dataloader)

        layers_idx = self.get_layers_idx()
        layers_dim = self.layers_dim
        num_layers = len(layers_dim)
        act_keys = self.get_act_keys()
        norm_diff = np.zeros((num_layers, ds_size))

        with torch.no_grad():
            for batch, (names, X, y) in enumerate(valid_dataloader):
                print(f"batch: {batch}/{ds_size}")
                X, y = X.to(self.device), y.to(self.device)
                self.model(X)
                for idx in range(num_layers):
                    act = self.activation[act_keys[idx]].\
                        detach().cpu().numpy()
                    norm_diff[idx, batch] = np.sum([np.linalg.norm(act[i] - act[0])
                                        for i in range(1, act.shape[0])])
 
        pickle.dump(norm_diff, open('../output/norm-diff.pkl', 'wb'))
        print(norm_diff)

    def timer_func(func):
        # This function shows the execution time of 
        # the function object passed
        def wrap_func(*args, **kwargs):
            t1 = time.time()
            result = func(*args, **kwargs)
            t2 = time.time()
            print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
            return result
        return wrap_func
  
    # @timer_func
    def get_activations(self, image, layers):
        # print("Calculating diff activations")
        layers_idx = self.get_layers_idx()
        layers_dim = self.layers_dim
        num_layers = len(layers_dim)
        act_keys = self.get_act_keys()
        act = []
        with torch.no_grad():
            self.model(image)
            for idx in layers:
                # act.append(self.activation[act_keys[idx]].detach().cpu().numpy())
                act.append(self.activation[act_keys[idx]])

        return act


    # @timer_func
    def get_last_activations(self, image, l):
        # print("Calculating diff activations")
        layers_idx = self.get_layers_idx()
        layers_dim = self.layers_dim
        num_layers = len(layers_dim)
        act_keys = self.get_act_keys()
        act = []
        with torch.no_grad():
            self.model(image)
            for idx in range(num_layers - l, num_layers):
                # act.append(self.activation[act_keys[idx]].detach().cpu().numpy())
                act.append(self.activation[act_keys[idx]])

        return act

    # @timer_func
    def get_first_activations(self, image, l):
        # print("Calculating diff activations")
        layers_idx = self.get_layers_idx()
        layers_dim = self.layers_dim
        num_layers = len(layers_dim)
        act_keys = self.get_act_keys()
        act = []
        with torch.no_grad():
            self.model(image)
            for idx in range(l):
                # act.append(self.activation[act_keys[idx]].detach().cpu().numpy())
                act.append(self.activation[act_keys[idx]])

        return act

    @timer_func
    def get_batch_diff_activations(self, s_model, clear_image, hazy_image):
        # print("Calculating diff activations")
        layers_idx = self.get_layers_idx()
        layers_dim = self.layers_dim
        num_layers = len(layers_dim)
        act_keys = self.get_act_keys()
        clear_act = []
        hazy_act = []
        diff = np.zeros(num_layers)
        with torch.no_grad():
            t1 = time.time()
            self.model(clear_image)
            for idx in range(num_layers):
                clear_act.append(self.activation[act_keys[idx]].detach().cpu().numpy())

            self.model(hazy_image)
            for idx in range(num_layers):
                hazy_act.append(self.activation[act_keys[idx]].detach().cpu().numpy())
            t2 = time.time()
            print(f'passing through network executed in {(t2-t1):.4f}s')

        t1 = time.time()
        for idx in range(num_layers):
            diff[idx] = np.mean([np.sqrt(np.linalg.norm(clear_act[idx] - hazy_act[idx]))
                                        for i in range(1, clear_act[idx].shape[0])])
        t2 = time.time()
        print(f'numpy operation executed in {(t2-t1):.4f}s')
        return diff

    def get_corrs(self):
        """ Compute the individual correlation
        Returns
        -------
        List of 2d tensors, each representing the connectivity between two
        consecutive layer.
        """
        self.model.eval()
        ds_size = len(self.dataloader.dataset)

        layers_idx = self.get_layers_idx()
        layers_dim = self.layers_dim
        num_layers = len(layers_dim)
        act_keys = self.get_act_keys()
        corrs = []

        for idx in range(num_layers - 1):
            logging.debug(f"working on layer {layers_idx[idx]} {str(act_keys[idx])[:18]}...")
            # prepare an array with the right dimension
            parent_arr = []
            child_arr = []

            with torch.no_grad():
                for batch, (X, y) in enumerate(self.dataloader):
                    X, y = X.to(self.device), y.to(self.device)
                    self.model(X)
                    parent_arr.append(self.activation[act_keys[idx]].\
                                  detach().cpu().numpy())
                    child_arr.append(self.activation[act_keys[idx + 1]].\
                                 detach().cpu().numpy())

            del self.activation[act_keys[idx]]
            self.hook_handles.pop(0)

            parent = np.vstack(parent_arr)
            parent = (parent - parent.mean(axis=0))
            parent /= np.abs(np.max(parent))
            child = np.vstack(child_arr)
            child = (child - child.mean(axis=0))
            child /= np.abs(np.max(child)) # child.std(axis=0)
            if np.any(np.isnan(parent)):
                print("nan in layer {layers_idx[idx]}")

            if np.any(np.isnan(child)):
                print("nan in layer {layers_idx[idx + 1]}")
            # corr = np.corrcoef(parent, child, rowvar=False)
            # x_len = corr.shape[0] // 2
            # y_len = corr.shape[1] // 2
            corr = utils.batch_mul(parent, child)
            logging.debug(f"correlation dimension: {corr.shape}, conn: {np.mean(corr):.6f}")
            corrs.append(corr)

        # print(corrs)
        self.model.train()
        return corrs

    def get_conns(self, corrs):
        conns = []
        for corr in corrs:
            conns.append(corr.mean())
        return conns

    def get_correlations(self):
        """ Compute the individual correlation
        Returns
        -------
        List of 2d tensors, each representing the connectivity between two
        consecutive layer.
        """
        self.model.eval()
        ds_size = len(self.dataloader.dataset)
        num_batch = len(self.dataloader)
        # params = list(self.model.parameters())

        layers_idx = self.get_layers_idx()
        layers_dim = self.layers_dim
        num_layers = len(layers_dim)
        act_keys = self.get_act_keys()

        corrs = [np.zeros((layers_dim[i][0], layers_dim[i + 1][0]))
                 for i in range(num_layers - 1)]

        act_means = [torch.zeros(layers_dim[i][0]).to(self.device)
                            for i in range(num_layers)]
        act_sq_sum = [torch.zeros(layers_dim[i][0]).to(self.device)
                         for i in range(num_layers)]
        act_max = torch.zeros(num_layers).to(self.device)

        with torch.no_grad():
            # Compute the mean of activations
            log.debug("Compute the mean and sd of activations")
            for batch, (X, y) in enumerate(self.dataloader):
                X, y = X.to(self.device), y.to(self.device)
                self.model(X)
                for i in range(num_layers):
                    act_means[i] += torch.sum(torch.nan_to_num(self.activation[act_keys[i]]),
                                              dim=0) 
                    act_sq_sum[i] += torch.sum(
                            torch.pow(torch.nan_to_num(self.activation[act_keys[i]]), 2), dim=0)
                    act_max[i] = abs(torch.max(act_max[i],
                                     abs(torch.max(self.activation[act_keys[i]]))))

            act_means = [act_means[i] / ds_size for i in range(num_layers)]
            act_sd = [torch.pow(act_sq_sum[i] / ds_size -
                                       torch.pow(act_means[i], 2), 0.5)
                             for i in range(num_layers)]

            # fix maximum activation for layers that are too close to zero
            for i in range(num_layers):
                sign = torch.sign(act_max[i])
                act_max[i] = sign * max(abs(act_max[i]), 0.001)
                act_sd[i] = torch.max(act_sd[i], 0.001 * \
                                      torch.ones(act_sd[i].shape).to(self.device))
                # logging.debug(f"nans in activation sd layer {i}: {torch.isnan(act_sd[i]).any()}")
                # logging.debug(f"nans in activation sd layer {i}: {torch.sum(torch.isnan(act_sd[i].view(-1)))}")
            # logging.debug(f"activation mean: {act_means}")
            # logging.debug(f"# nans in activation sd: {torch.nonzero(torch.isnan(act_sd.view(-1)))}")
            # logging.debug(f"activation max: {act_max}")

            for batch, (X, y) in enumerate(self.dataloader):
                # if batch % 100 == 0:
                    # log.debug(f"batch [{batch}/{num_batch}]")

                X, y = X.to(self.device), y.to(self.device)
                self.model(X)

                for i in range(num_layers - 1):
                    f0 = ((self.activation[act_keys[i]] - act_means[i]) /\
                          act_sd[i]).T
                    f1 = (self.activation[act_keys[i + 1]] - act_means[i + 1])/\
                          act_sd[i + 1]
                    corrs[i] += torch.matmul(f0, f1).detach().cpu().numpy()

        for i in range(num_layers - 1):
            corrs[i] = corrs[i] / ds_size # (layers_dim[i][0] * layers_dim[i + 1][0])
        self.model.train()
        return corrs

    def get_connectivity(self):
        """Find the connectivity of each layer, the mean of correlation matrix.
        """
        self.model.eval()
        ds_size = len(self.dataloader.dataset)
        num_batch = len(self.dataloader)
        # params = list(self.model.parameters())

        layers_dim = []
        hook_handles = []
        self.get_all_layers(self.model, layers_dim, hook_handles)
        num_layers = len(layers_dim)
        first_run = 1
        torch.set_printoptions(precision=4)

        corrs = [torch.zeros((layers_dim[i][0], layers_dim[i + 1][0])).to(self.device)
                 for i in range(num_layers - 1)]
        activation_means = [torch.zeros(layers_dim[i][0]).to(self.device)
                            for i in range(num_layers)]
        sq_sum = [torch.zeros(layers_dim[i][0]).to(self.device)
                         for i in range(num_layers)]
        act_max = [torch.zeros(layers_dim[i][0]).to(self.device)
                            for i in range(num_layers)]

        with torch.no_grad():
            # Compute the mean of activations
            log.debug("Compute the mean and sd of activations")
            for batch, (X, y) in enumerate(self.dataloader):
                # if batch % 100 == 0:
                    # log.debug(f"batch [{batch}/{num_batch}]")

                X, y = X.to(self.device), y.to(self.device)
                self.model(X)

                if first_run:
                    act_keys = list(self.activation.keys())
                    first_run = 0

                for i in range(num_layers):
                    activation_means[i] += torch.sum(
                        self.activation[act_keys[i]], dim=0)
                    sq_sum[i] += torch.sum(
                        torch.pow(self.activation[act_keys[i]], 2), dim=0)
                    act_max[i] = torch.max(self.activation[act_keys[i]])
            # log.debug("-------------------------------")

            activation_means = [activation_means[i] / ds_size
                                for i in range(num_layers)]
            activation_sd = [torch.pow(sq_sum[i] / ds_size -
                                       torch.pow(activation_means[i], 2), 0.5)
                             for i in range(num_layers)]

            # Compute normalized correlation
            log.debug("Compute normalized correlation")
            for batch, (X, y) in enumerate(self.dataloader):
                # if batch % 100 == 0:
                    # log.debug(f"batch [{batch}/{num_batch}]")

                X, y = X.to(self.device), y.to(self.device)
                self.model(X)

                # if first_run:
                #     act_keys = list(self.activation.keys())
                #     first_run = 0
                for i in range(num_layers - 1):
                    # Normalized activations
                    f0 = torch.div((self.activation[act_keys[i]] -
                                    activation_means[i]), act_max[i]).T
                    f1 = torch.div((self.activation[act_keys[i + 1]] -
                                    activation_means[i + 1]), act_max[i + 1])

                    # Zero-meaned activations
                    # f0 = (self.activation[act_keys[i]] - activation_means[i]).T
                    # f1 = (self.activation[act_keys[i + 1]] - activation_means[i + 1])

                    corrs[i] += torch.matmul(f0, f1)
        # Remove all hook handles
        for handle in hook_handles:
            handle.remove()

        self.model.train()
        return [torch.mean(corrs[i]).item()/(layers_dim[i][0] * layers_dim[i + 1][0])
                for i in range(num_layers - 1)]


  
def main():
    # preparing the hardware
    args = utils.get_args()
    device = utils.get_device(args)
    logger = utils.setup_logger()
    num_exper = 5

    data = Data(args.batch_size, C.DATA_DIR, args.dataset)
    num_classes = data.get_num_classes()
    train_dl, test_dl = data.train_dataloader, data.test_dataloader
    network = Network(device, args.arch, num_classes, args.pretrained)
    preprocess = network.preprocess
    model = network.set_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)


    corr = []
    test_acc = torch.zeros(num_exper)

    for i in range(num_exper):
        logger.debug("=" * 10 + " experiment " + str(i + 1) + "=" * 10)
        train_acc, _ = train(model, train_dl, loss_fn, optimizer,
                             args.train_epochs, device)
        test_acc[i] = test(model, test_dl, loss_fn, device)

        activations = Activations(model, test_dl, device, args.batch_size)
        # corr.append(activations.get_connectivity())
        corrs = activations.get_corrs()
        my_corrs = activations.get_correlations()
        conns = activations.get_conns(corrs)
        my_conns = activations.get_conns(my_corrs)

        diff = [np.sum(np.abs(corrs[i] - my_corrs[i])) for i in
                range(len(corrs))]
        print(diff)
        diff_cons = np.array(conns) - np.array(my_conns)
        print(diff_cons)
        corr.append(corrs)

        utils.save_model(model, C.OUTPUT_DIR, args.arch + f'-{i}-model.pt')
        logger.debug('model is saved...!')

        utils.save_vars(test_acc=test_acc, corr=corr)

    print(corr)
    plot_tool.plot_connectivity(test_acc, corr)


if __name__ == '__main__':
    main()
