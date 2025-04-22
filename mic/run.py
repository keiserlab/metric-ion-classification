# python script to automatically run classification and output identities for densities of interest
import argparse
import logging
import multiprocessing
from tqdm import tqdm
import numpy as np
import torch

from mic.fingerprints.FPParser import FPParser
from mic.models.MIC import MIC

mic_logger = logging.getLogger('mic_logger')

def get_argparser():
    parser = argparse.ArgumentParser(
        prog="Metric Ion Classification",
        description="Runs MIC for provided PDB file and densities, if specified")
    parser.add_argument('pdb_file', metavar='pdb_file', type=str, help='path to protein file to evaluate')
    parser.add_argument('-e', '--entries', action='store', type=str)
    parser.add_argument('-n', '--num_processes', action='store', default=8, type=int)
    parser.add_argument('-t', '--fp_type', action='store', default='prune-eifp', 
                        choices=['prune-eifp'])
    parser.add_argument('-o', '--outfile', action='store')
    parser.add_argument('-p', '--preds_only', action='store_true')
    parser.add_argument('-ext', '--extended_set', action='store_true')
    parser.add_argument('-h', '--hetatm', action='store_true')
    parser.add_argument('-cpu', '--cpu_only', action='store_true')
    parser.add_argument('-f', '--fps_only', action='store_true', help='Generate fingerprints, no predicting.')
    parser.add_argument('-fd', '--flag_distance', type=float, default=4)   
    parser.add_argument('-ft', '--flag_threshold', type=int, default=3)
 
    parser.add_argument('-l', '--length', help='desired fp length, suggested: 4096', type=int, default=4096)
    parser.add_argument('-sn', '--shell_number', type=int, default=18)
    parser.add_argument('-sr', '--shell_radius', type=float, default=0.25)
    parser.add_argument('-ub', '--unblinded', action='store_true')
    parser.add_argument('-co', '--symexp_cutoff', default=7, type=int)
    parser.add_argument('-b', '--bit', action='store_true')
    return parser

def main():
    parser = get_argparser()
    args = parser.parse_args()
   
 
    fp_type = args.fp_type
 
    torch_avail = torch.cuda.is_available()
    if not torch_avail or args.cpu_only:
        device = 'cpu'
    else:
        device = 'cuda' 

    mic_logger.info(f"Running with settings:\nPDB File: {args.pdb_file}\nFP Type: {fp_type}\nAll classes: {args.extended_set}")
    mic_logger.info(f'Running on GPU: {device=="cuda"}')

    fpparser = FPParser(args.pdb_file, fp_type,
                        length = args.length,
                        shell_number = args.shell_number,
                        shell_radius = args.shell_radius,
                        blind = not args.unblinded,
                        symexp_cutoff = args.symexp_cutoff, 
                        count = not args.bit,
                        entry_file = args.entries,
                        n_proc = args.num_processes,
                        flag_distance = args.flag_distance,
                        flag_threshold = args.flag_threshold)

    if len(fpparser.errors) > 0:
        mic_logger.warning(f"{len(fpparser.errors)} sites failed to generate fingerprints, check input file.")
    
    if not args.fps_only:
        model = MIC(fp_type, args.extended_set,args.hetatm, device=device)
        results = model.predict(fpparser.fps, fpparser.format_entries(), return_proba = not args.preds_only)
        results['flag_cluster'] = fpparser.flags    

        if args.outfile is not None:
            results.to_csv(args.outfile)
                        
        else:
            print(results)

    else:
        fpparser.get_as_dataframe().to_csv(args.outfile, header=None)
        mic_logger.info(f'{fpparser.get_as_dataframe().shape[0]} fingerprints written to {args.outfile}.')    

if __name__ == '__main__':
    main()
