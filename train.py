import os
import torch
import time
import matplotlib.pyplot as plt
from dataset import get_loader
from dagmm import DAGMM
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR

def plot_loss_moment(losses,hyp):
    _, ax = plt.subplots(figsize=(16, 9), dpi=80)
    ax.plot(losses, 'blue', label='train', linewidth=1)
    ax.set_title('Loss change in training')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iteration')
    ax.legend(loc='upper right')
    plt.savefig(os.path.join(hyp['img_dir'], 'loss_dagmm.png'))
    
    
device = torch.device("cuda:0")
def train(hyp):
 
    train_loader , _= get_loader(hyp,mode = 'train')
    model = DAGMM(hyp)
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(),hyp['lr'], amsgrad=True)
    scheduler = MultiStepLR(optim, [5, 8], 0.1)
#     iter_wrapper = lambda x: tqdm(x, total=len(train_loader))
    
    loss_total = 0
    recon_error_total = 0
    e_total = 0
    loss_plot = []
    start_time = time.time()
    
    for epoch in range(hyp['epochs']):
        for i, (input_data, labels) in enumerate(train_loader):
            input_data = input_data.to(device)
            model.train()
            optim.zero_grad()

            enc,dec,z,gamma = model(input_data)
            input_data,dec,z,gamma = input_data.cpu(),dec.cpu(),z.cpu(),gamma.cpu()
            loss, recon_error, e, p = model.loss_func(input_data, dec, gamma, z)
#             print('loss',loss,'recon_error',recon_error,'e',e,'p',p)
     
         
            
            loss_total += loss.item()
            recon_error_total += recon_error.item()
            e_total += e.item()

#             model.zero_grad()

            loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optim.step()
            

            if (i+1) % hyp['print_iter'] == 0:
                elapsed = time.time() - start_time
                
                log = "Time {:.2f}, Epoch [{}/{}], Iter [{}/{}], lr {} ".format(elapsed, epoch+1, hyp['epochs'], i+1, len(train_loader), optim.param_groups[0]['lr'])
                
                log+= 'loss {:.4f}, recon_error {:.4f}, energy {:.4f} '.format(loss_total/hyp['print_iter'],
                                                                   recon_error/hyp['print_iter'],e_total/hyp['print_iter'])
                loss_plot.append(loss_total/hyp['print_iter'])
                loss_total = 0
                recon_error_total = 0 
                e_total = 0
                print(log)
        
           
        if (epoch+1) % hyp['savestep_epoch'] == 0:
                torch.save(model.state_dict(),
                    os.path.join(hyp['save_path'], '{}_dagmm.pth'.format(epoch)))
                
        scheduler.step()
                
    plot_loss_moment(loss_plot,hyp)

if __name__ == "__main__":
	hyp={
	 'input_dim':118,
	 'hidden1_dim':60,
	 'hidden2_dim':30,
	 'hidden3_dim':10,
	 'zc_dim':1,
	 'emb_dim':10,
	 'n_gmm':2,
	 'dropout':0.5,
	 'lambda1':0.1,
	 'lambda2':0.005,
	 'lr' :1e-4,
	 'batch_size':100,
	 'epochs': 10,
	 'print_iter':300,
	 'savestep_epoch': 2,
	 'save_path': './models/',
	 'data_dir': '../dagmm-master/kdd_cup.npz',
	 'img_dir': './result/',
	 'ratio' : 0.8
	}

	if not os.path.isdir('./models/'):
		os.mkdir('./models/')
	if not os.path.isdir('./result/'):
		os.mkdir('./result/')
		
	train(hyp)