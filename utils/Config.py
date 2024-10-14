def load_net_para(data_name):
    L_dims = None
    latent_dim = None
    args = {
        'constant': {},
        'Use_cuda': False,
        'device': 'cpu'
    }
    if data_name == 'Yale15_3_165':
        dims1 = [4096, 1024]  #
        dims2 = [3304, 1024]  #
        dims3 = [6750, 1024]  #
        L_dims = [dims1, dims2, dims3]
        latent_dim = 128
        args['constant']['p_Batch_size'] = 128
        args['constant']['Batch_size'] = 128
        args['constant']['show_result_epoch'] = 50
        args['constant']['p_epochs'] = 500
        args['constant']['epochs'] = 500
    elif data_name == 'leaves100_3_1600':
        dims1 = [64, 1024]  # 1600*64
        dims2 = [64, 1024]  # 1600*64
        dims3 = [64, 1024]  # 1600*64
        L_dims = [dims1, dims2, dims3]
        latent_dim = 128  # 256
        args['constant']['p_Batch_size'] = 128
        args['constant']['Batch_size'] = 128
        args['constant']['show_result_epoch'] = 50
        args['constant']['p_epochs'] = 500
        args['constant']['epochs'] = 500
    elif data_name == 'ORL40_3_400':
        dims1 = [4096, 1024]  #
        dims2 = [3304, 1024]  #
        dims3 = [6750, 1024]  #
        L_dims = [dims1, dims2, dims3]
        latent_dim = 128
        args['constant']['p_Batch_size'] = 128
        args['constant']['Batch_size'] = 128
        args['constant']['show_loss_step'] = 100
        args['constant']['show_result_epoch'] = 50
        args['constant']['p_epochs'] = 500
        args['constant']['epochs'] = 351
    elif data_name == 'YaleB10_3_650':
        dims1 = [2500, 1024]  #
        dims2 = [3304, 1024]  #
        dims3 = [6750, 1024]  #
        L_dims = [dims1, dims2, dims3]
        latent_dim = 128  # 256
        args['constant']['p_Batch_size'] = 128
        args['constant']['Batch_size'] = 128
        args['constant']['show_loss_step'] = 100
        args['constant']['show_result_epoch'] = 10
        args['constant']['p_epochs'] = 500
        args['constant']['epochs'] = 500
    elif data_name == 'Reuters_dim10':
        dims1 = [10, 1024]  # 18758*10  # 100
        dims2 = [10, 1024]  # 18758*10
        L_dims = [dims1, dims2]
        latent_dim = 128  # 30
        args['constant']['p_Batch_size'] = 128 # 32
        args['constant']['Batch_size'] = 128
        args['constant']['show_loss_step'] = 100
        args['constant']['show_result_epoch'] = 20
        args['constant']['p_epochs'] = 500
        args['constant']['epochs'] = 500
    if data_name == 'ALOI100_4_10800':
        dims1 = [77, 1024]  # 10800*77 100
        dims2 = [13, 1024]  # 10800*13  100
        dims3 = [64, 1024]  # 10800*64  100
        dims4 = [125, 1024]  # 10800*125  100
        L_dims = [dims1, dims2, dims3, dims4]
        latent_dim = 128  # 100
        args['constant']['p_Batch_size'] = 128  # 32
        args['constant']['Batch_size'] = 128
        args['constant']['show_loss_step'] = 100
        args['constant']['show_result_epoch'] = 20
        args['constant']['p_epochs'] = 500  # 200
        args['constant']['epochs'] = 500
    return L_dims, latent_dim, args