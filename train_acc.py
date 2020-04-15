import time
import copy
import torch
from options.base_options import BaseOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    # training dataset
    opt = BaseOptions().parse()
    print('>>>>>lambda_L1:', opt.lambda_L1)
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    opt.dataset_size = dataset_size
    
    print('#training images = %d' % dataset_size)

    # model
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        visualizer.reset()

        epoch_iter = 0 # iterator within an epoch
        epoch_loss = 0.0
        model.reset_gradients()
        for i, data in enumerate(dataset):          
            model.set_input(data)
            model.compute_gradients(epoch, epoch_iter)
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            epoch_loss += model.loss_G_total.detach().cpu()
        epoch_loss /= dataset_size
        model.step_gradients()
        model.update_learning_rate()
        
        losses = {"EpochLoss": epoch_loss}
        visualizer.plot_current_losses(epoch, 0.0, opt, losses)

        if epoch % opt.save_epoch_freq == 0:
            model.WriteTextureToFile(opt.results_dir + 'texture_'+str(epoch)+'.png')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
