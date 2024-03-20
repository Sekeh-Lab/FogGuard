 
    # train_dataset = ds.CustomImageDataset(img_dir="/home/soheil/data/coco/images/train2014/",
    #                                       transform=transform,
    #                                       fog_level=35)

    # eval_dataset = ds.CustomImageDataset(img_dir="/home/soheil/data/coco/images/val2014/",
    #                                       transform=transform,
    #                                       fog_level=35)

    # train_dl = torch.utils.data.DataLoader(dataset = train_dataset,
    #                                     batch_size = batch_size,
    #                                     # collate_fn=lambda batch: tuple(zip(*batch)),
    #                                     )

    # eval_dl = torch.utils.data.DataLoader(dataset = eval_dataset,
    #                                     batch_size = batch_size,
    #                                     # collate_fn=lambda batch: tuple(zip(*batch)),
    #                                     )

    # Load training dataloader
    # train_dl = ds._create_data_loader(
    #     train_path,
    #     mini_batch_size,
    #     t_model.hyperparams['height'],
    #     args.n_cpu,
    #     transform=AUGMENTATION_TRANSFORMS,
    #     # num_samples=1000,
    #     multiscale_training=args.multiscale_training)

    # valid_dl = ds._create_data_loader(
    #     valid_path,
    #     mini_batch_size,
    #     t_model.hyperparams['height'],
    #     args.n_cpu,
    #     transform=AUGMENTATION_TRANSFORMS,
    #     # num_samples=100,
    #     multiscale_training=args.multiscale_training)

