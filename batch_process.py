import subprocess

def batch_process():
    n_classes = [4, 6, 8, 10]
    dim_capsules = [8, 16, 24]
    for n_class in n_classes:
        for dim_caps in dim_capsules:
            args = ['python3.6']
            args.append('capsulenet.py')
            args.append('--batch_dir')
            args.append('./data')
            args.append('--n_class') 
            args.append(str(n_class))
            args.append('--dim_capsule')
            args.append(str(dim_caps))
            args.append('--save_dir')
            args.append('./results/'+'n_class'+str(n_class)+'dim'+str(dim_caps))
            subprocess.call(args)

batch_process()
