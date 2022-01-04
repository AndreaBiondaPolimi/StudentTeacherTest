import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser
from einops import rearrange, reduce
from AnomalyNet import AnomalyNet
from AnomalyDataset import AnomalyDataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from utils import increment_mean_and_var, load_model
from sklearn.metrics import roc_curve, auc
from scipy import integrate 
from Metrics import get_ovr, visualize_results, preprocess_data, bg_mask, batch_evaluation, get_performance, get_roc, image_evaluation, get_iou

def parse_arguments():
    parser = ArgumentParser()

    # program arguments
    parser.add_argument('--dataset', type=str, default='grid', help="Dataset to infer on (in data folder)")
    parser.add_argument('--test_size', type=int, default=20, help="Number of batch for the test set")
    parser.add_argument('--n_students', type=int, default=3, help="Number of students network to use")
    parser.add_argument('--patch_size', type=int, default=33, choices=[17, 33, 65])
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--visualize', type=bool, default=True, help="Display anomaly map batch per batch")

    # trainer arguments
    parser.add_argument('--gpus', type=int, default=(1 if torch.cuda.is_available() else 0))
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    return args


def get_error_map(students_pred, teacher_pred):
    # student: (batch, student_id, h, w, vector)
    # teacher: (batch, h, w, vector)
    mu_students = reduce(students_pred, 'b id h w vec -> b h w vec', 'mean')
    err = reduce((mu_students - teacher_pred)**2, 'b h w vec -> b h w', 'sum')
    return err


def get_variance_map(students_pred):
    # student: (batch, student_id, h, w, vector)
    sse = reduce(students_pred**2, 'b id h w vec -> b id h w', 'sum')
    msse = reduce(sse, 'b id h w -> b h w', 'mean')
    mu_students = reduce(students_pred, 'b id h w vec -> b h w vec', 'mean')
    var = msse - reduce(mu_students**2, 'b h w vec -> b h w', 'sum')
    return var


@torch.no_grad()
def calibrate(teacher, students, dataloader, device):
    print('calibrating teacher on Student dataset.')
    t_mu, t_var, t_N = 0, 0, 0
    for _, batch in tqdm(enumerate(dataloader)):
        inputs = batch['image'].to(device)
        t_out = teacher.fdfe(inputs)
        t_mu, t_var, t_N = increment_mean_and_var(t_mu, t_var, t_N, t_out)
    
    print('calibrating scoring parameters on Student dataset.')
    max_err, max_var = 0, 0
    mu_err, var_err, N_err = 0, 0, 0
    mu_var, var_var, N_var = 0, 0, 0

    for _, batch in tqdm(enumerate(dataloader)):
        inputs = batch['image'].to(device)

        t_out = (teacher.fdfe(inputs) - t_mu) / torch.sqrt(t_var)
        s_out = torch.stack([student.fdfe(inputs) for student in students], dim=1)

        s_err = get_error_map(s_out, t_out)
        s_var = get_variance_map(s_out)
        mu_err, var_err, N_err = increment_mean_and_var(mu_err, var_err, N_err, s_err)
        mu_var, var_var, N_var = increment_mean_and_var(mu_var, var_var, N_var, s_var)

        max_err = max(max_err, torch.max(s_err))
        max_var = max(max_var, torch.max(s_var))
    
    return {"teacher": {"mu": t_mu, "var": t_var},
            "students": {"err":
                            {"mu": mu_err, "var": var_err, "max": max_err},
                         "var":
                            {"mu": mu_var, "var": var_var, "max": max_var}
                        }
            }


@torch.no_grad()
def get_score_map(inputs, teacher, students, params):
    t_out = (teacher.fdfe(inputs) - params['teacher']['mu']) / torch.sqrt(params['teacher']['var'])
    s_out = torch.stack([student.fdfe(inputs) for student in students], dim=1)

    s_err = get_error_map(s_out, t_out)
    s_var = get_variance_map(s_out)
    score_map = (s_err - params['students']['err']['mu']) / torch.sqrt(params['students']['err']['var'])\
                    + (s_var - params['students']['var']['mu']) / torch.sqrt(params['students']['var']['var'])
    
    return score_map


def visualize(img, gt, score_map, max_score):
    plt.figure(figsize=(13, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f'Original image')

    plt.subplot(1, 3, 2)
    plt.imshow(gt, cmap='gray')
    plt.title(f'Ground thuth anomaly')

    plt.subplot(1, 3, 3)
    plt.imshow(score_map, cmap='jet')
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.imshow(score_map, cmap='jet', alpha=0.5, interpolation='none')
    plt.colorbar(extend='both')
    plt.title('Anomaly map')

    plt.clim(0, max_score)
    plt.show(block=True)


def detect_anomaly(args):
    # Choosing device 
    device = torch.device(f"cuda:{args.cuda}" if args.gpus else "cpu")
    print(f'Device used: {device}')

    # Teacher network
    teacher = AnomalyNet.create((args.patch_size, args.patch_size))
    teacher.eval().to(device)

    # Load teacher model
    load_model(teacher, f'../model/{args.dataset}/teacher_{args.patch_size}_net.pt')

    # Students networks
    students = [AnomalyNet.create((args.patch_size, args.patch_size)) for _ in range(args.n_students)]
    students = [student.eval().to(device) for student in students]

    # Loading students models
    for i in range(args.n_students):
        model_name = f'../model/{args.dataset}/student_{args.patch_size}_net_{i}.pt'
        load_model(students[i], model_name)

    if (args.dataset == 'grid'):
        tr = transforms.Compose([
                            transforms.Resize((args.image_size, args.image_size)),
                            transforms.Grayscale(num_output_channels=3),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
         tr = transforms.Compose([
                            transforms.Resize((args.image_size, args.image_size)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    # calibration on anomaly-free dataset
    calib_dataset = AnomalyDataset(root_dir=f'../data/{args.dataset}',
                                    transform=tr,
                                    type='train',
                                    label=0)

    calib_dataloader = DataLoader(calib_dataset, 
                                   batch_size=args.batch_size, 
                                   shuffle=False, 
                                   num_workers=args.num_workers)
    import os
    import pickle
    param_file = f'../model/{args.dataset}/params_{args.patch_size}_{args.image_size}'
    if (os.path.isfile(param_file)):
        with open(param_file, 'rb') as f:
            params = pickle.load(f)
    else:
        params = calibrate(teacher, students, calib_dataloader, device)
        with open(param_file, 'wb') as f:
            pickle.dump(params, f)


    if (args.dataset == 'grid'):
        tr = transforms.Compose([
                    transforms.Resize((args.image_size, args.image_size)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
         tr = transforms.Compose([
                    transforms.Resize((args.image_size, args.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    # Load testing data
    test_dataset = AnomalyDataset(root_dir=f'../data/{args.dataset}',
                                  transform=tr,
                                  gt_transform=transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor()]),
                                  type='test')

    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=args.batch_size, 
                                 shuffle=False, 
                                 num_workers=args.num_workers)

    
    # Build anomaly map
    y_score = np.array([])
    y_true = np.array([])
    test_iter = iter(test_dataloader)

    avg_au_roc = 0; avg_au_iou = 0; avg_au_pro = 0
    for i in range(args.test_size):
        batch = next(test_iter)
        inputs = batch['image'].to(device)
        gt = batch['gt'].cpu()

        score_map = get_score_map(inputs, teacher, students, params).cpu()
        y_score = np.concatenate((y_score, rearrange(score_map, 'b h w -> (b h w)').numpy()))
        y_true = np.concatenate((y_true, rearrange(gt, 'b c h w -> (b c h w)').numpy()))

        
        unorm = transforms.Normalize((-1, -1, -1), (2, 2, 2)) # get back to original image
        max_score = (params['students']['err']['max'] - params['students']['err']['mu']) / torch.sqrt(params['students']['err']['var'])\
            + (params['students']['var']['max'] - params['students']['var']['mu']) / torch.sqrt(params['students']['var']['var']).item()
        img_in = rearrange(unorm(inputs).cpu(), 'b c h w -> b h w c')
        gt_in = rearrange(gt, 'b c h w -> b h w c')

        for b in range(args.batch_size):
            if args.visualize:
                visualize(img_in[b, :, :, :].squeeze(), 
                        gt_in[b, :, :, :].squeeze(), 
                        score_map[b, :, :].squeeze(), 
                        max_score)

            res_score = score_map[b, :, :].squeeze().numpy()
            res_gt = gt_in[b, :, :].squeeze().numpy()
            res_gt[res_gt > 0] = 1

            #plt.imshow(res_score)
            #plt.show()
            #plt.imshow(res_gt)
            #plt.show()

            step = 0.1
            results = []
            for tresh in np.arange (0, 80, step):
                results.append (compute_performance({'tresh': tresh.copy(), 'residual': res_score.copy(), 'valid_gt': res_gt.copy()}))
                


            #Compute roc,auc and iou scores async
            tprs = []; fprs = []; ious = [] ;ovrs = []
            """
            args = [{'tresh': tresh.copy(), 'residual': res_score.copy(), 'valid_gt': res_gt.copy()} for tresh in np.arange (0.1, 0.3, step)] 
            with Pool(processes=2) as pool:  # multiprocessing.cpu_count()
                results = pool.map(compute_performance, args, chunksize=1)
            """

            for result in results:
                tpr = result['tpr']; fpr = result['fpr']; iou = result['iou']; ovr = result['ovr']
                #print (fpr, tpr, iou, ovr)
                if (fpr <= 0.3):
                    #print (tpr, fpr, iou)
                    tprs.append(tpr); fprs.append(fpr); ious.append(iou); ovrs.append(ovr)

            if (len(fprs) > 0):
                tprs = np.array(tprs); fprs = np.array(fprs); ious = np.array(ious); ovrs = np.array(ovrs)
                au_roc = (-1 * integrate.trapz(tprs, fprs))/(np.max(fprs)*np.max(tprs))
                #au_roc = (-1 * integrate.trapz(tprs, fprs))/(np.max(fprs))
                #au_iou = (-1 * integrate.trapz(ious, fprs))/(np.max(fprs)*np.max(ious))
                au_iou = (-1 * integrate.trapz(ious, fprs))/(np.max(fprs))
                au_pro = (-1 * integrate.trapz(ovrs, fprs))/(np.max(fprs)*np.max(ovrs))
                #au_pro = (-1 * integrate.trapz(ovrs, fprs))/(np.max(fprs))
            else:
                au_iou = 0; au_roc=0; au_pro=0

            print ("COUNT FPR: ", len(fprs))
            #print ("MAX FPR: ", np.max(fprs))
            print ("Area under ROC:", au_roc)
            print ("Area under IOU:", au_iou)  
            print ("Area under PRO:", au_pro) 

            avg_au_roc += au_roc
            avg_au_iou += au_iou
            avg_au_pro += au_pro
    
    print ("MEAN Area under ROC:", avg_au_roc/(args.test_size))
    print ("MEAN Area under IOU:", avg_au_iou/(args.test_size))
    print ("MEAN Area under PRO:", avg_au_pro/(args.test_size))
        



def compute_performance(args):
    tresh = args['tresh']
    residual = args['residual']
    valid_gt = args['valid_gt']

    residual[residual < tresh] = 0
    residual[residual >= tresh] = 1
    
    #plt.imshow(residual)
    #plt.show()

    tpr, fpr = get_roc(valid_gt, residual)
    iou = get_iou(valid_gt, residual)
    ovr = get_ovr(valid_gt, residual)

    tresh = None; residual = None; valid_gt = None

    return {'tpr': tpr, 'fpr': fpr, 'iou': iou, 'ovr': ovr}




if __name__ == '__main__':
    args = parse_arguments()
    
    
    detect_anomaly(args)


    #detect_anomaly(args, True)
