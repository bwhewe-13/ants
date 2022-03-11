########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

from ants.transport import Transport as load

import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--file', action='store', type=str, dest='file', \
                    help="Path to Input File")
parser.add_argument('--save', action='store_true', dest='save_flux', \
                    help="Use if saving the calculated flux")
parser.add_argument('--graph', action='store_true', dest='graph_flux', \
                    help="Use if graphing the calculated flux")
args = parser.parse_args()

problem = load(args.file)
if args.graph_flux and args.save_flux:
    problem.run_graph_save()
elif args.graph_flux:
    problem.run_graph()
elif args.save_flux:
    problem.run_save()
else:
    problem.run()


