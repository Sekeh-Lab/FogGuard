
class VOC(Dataset):
    def __init__(self, data_src, data_dst, data_type, classes, transform,
                 img_size, batch_size, num_samples=None, copy_file=False):
        """Create yolo compatible voc directory
        Parameters
        ----------
        self: type
            description
        data_src: str
            the path to the original VOC dataset
        data_dst: str
            the path to the destination yolo compatible voc dataset
        """
        self.classes = classes
        self.data_type = data_type
        self.data_src = data_src
        self.data_dst = data_dst
        self.transform = transform
        self.img_size = img_size
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.image_inds = []
        ds_dirs = os.listdir(os.path.join(self.data_src, "VOCdevkit"))
        for ds_dir in ds_dirs:
            img_inds_file = os.path.join(self.data_src, "VOCdevkit", ds_dir,
                                             'ImageSets', 'Main',
                                             self.data_type + '.txt')
            with open(img_inds_file, 'r') as f:
                txt = f.readlines()
                self.image_inds += [(os.path.join("VOCdevkit", ds_dir), line.strip())
                                    for line in txt]

        self.create_folders()
        if copy_file:
            self.create_files()

    def create_folders(self):
        image_dst = os.path.join(self.data_dst, self.data_type, 'images')
        label_dst = os.path.join(self.data_dst, self.data_type, 'labels')
        self.make_folders(image_dst)
        self.make_folders(label_dst)
        
    def make_folders(self, path):
        """Create directory if not exists"""
        if not os.path.exists(path):
            # shutil.rmtree(path)
            os.makedirs(path)

    def replace_folders(self, path):
        """Replace the directory if exists"""
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    def __len__(self):
        return len(self.image_inds)

    def create_files(self, use_difficult_bbox=False):
        image_list = ""
        for image_ind in tqdm(self.image_inds):
            image_path = os.path.join(self.data_src, image_ind[0], 'JPEGImages',
                                      image_ind[1] + '.jpg')
            # save the file in new diretory
            image_dst = os.path.join(self.data_dst, self.data_type, 'images',
                                    image_ind[1] + '.jpg')
            image_list += image_dst + "\n"
            shutil.copyfile(image_path, image_dst)
            annotation = image_path
            anno_path = os.path.join(self.data_dst, self.data_type, 'labels',
                                     image_ind[1] + '.txt')
            label_path = os.path.join(self.data_src, image_ind[0], 'Annotations',
                                      image_ind[1] + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            width = int(root.find("size").find("width").text)
            height = int(root.find("size").find("height").text)
            yolo_annot = ""
            for obj in objects:
                difficult = obj.find('difficult').text.strip()
                if (not use_difficult_bbox) and(int(difficult) == 1):
                    continue
                bbox = obj.find('bndbox')
                if obj.find('name').text.lower().strip() in self.classes:
                    class_ind = self.classes.index(obj.find('name').text.lower().strip())
                    xmin = int(bbox.find('xmin').text.strip())
                    xmax = int(bbox.find('xmax').text.strip())
                    ymin = int(bbox.find('ymin').text.strip())
                    ymax = int(bbox.find('ymax').text.strip())
                    x, y, w, h = self.xml_to_yolo_bbox([xmin, ymin, xmax, ymax],
                                                    width, height)
                    # annotation += ' ' + ','.join([xmin, ymin, xmax, ymax,
                    # str(class_ind)])
                    yolo_annot += f"{class_ind} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                    # f.write(f"{class_ind} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            # print(yolo_annot)
            open(anno_path, 'w').write(yolo_annot)
        open(os.path.join(self.data_dst, self.data_type, "list"), 'w').write(image_list)

    def create_dataloader(self):
        dataset = ListDataset(
            list_path=os.path.join(self.data_dst, self.data_type, "list"),
            img_size=self.img_size,
            multiscale=False,
            transform=self.transform,
            num_samples=self.num_samples
        )

        # sampler = BatchSampler(RandomSampler(dataset, replacement=True),
        #                          self.batch_size, drop_last=True)
        sampler = RandomSampler(dataset, replacement=True,
                                num_samples=self.num_samples)

        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            # shuffle=True,
            # num_workers=n_cpu,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            worker_init_fn=utils.worker_seed_set
        )

        return dataloader

    def xml_to_yolo_bbox(self, bbox, w, h):
        # xmin, ymin, xmax, ymax
        x_center = ((bbox[2] + bbox[0]) / 2) / w
        y_center = ((bbox[3] + bbox[1]) / 2) / h
        width = (bbox[2] - bbox[0]) / w
        height = (bbox[3] - bbox[1]) / h
        return [x_center, y_center, width, height]


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None,
                 path=None, fog_level=None):
        # self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_dir_list = [os.path.join(img_dir, item.name)
                             for item in pathlib.Path(img_dir).glob("*.jpg")
                             if not item.is_dir()][:100]
        self.transform = transform
        self.target_transform = target_transform
        self.image_list = []
        self.fog_level = fog_level
        self.load_images()


    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, idx):
        images = self.image_list[idx]
        trainsformed_image_list = []
        # image = Image.open(self.img_dir_list[idx]).convert('RGB')
        # image = read_image(self.img_dir_list[idx])
        if self.transform:
            for image in images:
                trainsformed_image_list.append(self.transform(image))
        if self.target_transform:
            label = self.target_transform(label)

        return tuple(trainsformed_image_list)


    def load_images(self):
        image_list = []
        ds_dir = pathlib.Path(__file__).parents[2] / 'data/torch-dataset.pt'
        if os.path.exists(ds_dir):
            image_list = torch.load(ds_dir)
        else:
            print("Start loading images")
            for img in tqdm(self.img_dir_list):
                image = Image.open(img).convert('RGB')

                if self.fog_level:
                    image_numpy = np.array(image)
                    
                    foggy_image_list = self.foggify_image(image_numpy)
                    image_list.append(tuple([image] + foggy_image_list))
                else:
                    image_list.append((image))

            torch.save(image_list, ds_dir)
        self.image_list = image_list


    def foggify_image(self, image):
        image_list = []
        A = 0.5  
        for i in range(self.fog_level):
            beta = 0.01 * i
            hazy_img_numpy = self.add_haz(image, beta)
            image_list.append(Image.fromarray(hazy_img_numpy))
        return image_list


    # @jit
    def add_haz(self, img_f, beta):
        img_f = img_f / 255
        (row, col, chs) = img_f.shape
        center = (row // 2, col // 2)  
        size = math.sqrt(max(row, col)) 

        x, y = np.meshgrid(np.linspace(0, row, row, dtype=int),
                        np.linspace(0, col, col, dtype=int))
        d = -0.04 * np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) + size
        d = np.tile(d, (3, 1, 1)).T
        trans = np.exp(-d * beta)

        # A = 255
        A = 0.5
        hazy = img_f * trans + A * (1 - trans)
        # hazy = np.array(hazy, dtype=np.uint8)

        return hazy

    # def add_haz_0(img_f, beta):
    #     (row, col, chs) = img_f.shape
    #     center = (row // 2, col // 2)  
    #     size = math.sqrt(max(row, col)) 
    #     A = 0.5  

    #     for j in range(row):
    #         for l in range(col):
    #             d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
    #             td = math.exp(-beta * d)
    #             img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
    #     return img_f


class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True,
                 transform=None, num_samples=None):
        with open(list_path, "r") as file:
            # if num_samples:
            #     self.img_files = file.readlines()[:num_samples]
            # else:
            self.img_files = file.readlines()

        self.label_files = []
        for path in self.img_files:
            image_dir = os.path.dirname(path)
            label_dir = "labels".join(image_dir.rsplit("images", 1))
            assert label_dir != image_dir, \
                f"Image path must contain a folder named 'images'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform
        self.num_samples = num_samples if num_samples else len(self.img_files)

    def __getitem__(self, index):

        #  Image
        try:
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)

        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        #  Label
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)

        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        #  Transform
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print("Could not apply transform.")
                return

        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)


    def load_annotations(self, annot_path):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        print('###################the total image:', len(annotations))
        return annotations

def load_coco_detection_dataset(**kwargs):
    torchvision.disable_beta_transforms_warning()
    root = pathlib.Path("/home/soheil/data") / "coco"
    return datasets.CocoDetection(str(root / "images" / "train2014"),
                                  str(root / "annotations" / "instances_train2014.json"),
                                  **kwargs)

