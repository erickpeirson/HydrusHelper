# Usage:
#  $ python process_tlevel.py [path to data] [path to output]
#
# [path to data] should point to a directory containing Hydrus workspaces (subdirectories)
#  each of which contains a file called "T_Level.out". If such a file is not encountered
#  in a subdirectory, that subdirectory will be ignored quietly.
#

import sys
import os
import pandas
import re

hpath = sys.argv[1]
outpath = sys.argv[2]

# If the output directory already doesn't exist, it will be created.
if not os.path.exists(outpath):
    os.makedirs(outpath)

# Each output file will be named according to the workspace from which it was generated.    
outfile = '{name}.csv'

for name in os.listdir(hpath):
    tpath = os.path.join(hpath, name, 'T_Level.out')
    
    # Check for T_Level.out. If it's not there, move on quietly.
    if os.path.exists(tpath):
        print 'reading data in', name, '...',
        with open(tpath, 'r') as f:   # Hydrus data files are a mess.
            raw = f.readlines()
        print 'done'
        
        data = []
        for line in raw[6:]:                   # The first several lines are junk headers.
            data.append(re.split('\s+', line)) # Data are whitespace delimited.
            
        df = pandas.DataFrame(data[3:-1], columns=data[0])
        
        # R doesn't like parentheses in variable names.
        df['E'] = df['sum(Evap)']
        
        # TODO: make the selected parameters configurable.
        selected = df[['rRoot', 'vRoot', 'rTop', 'vTop', 'E']]
        
        print 'writing data to output directory...',
        # Ignore the "index" column, which Pandas generated automatically.
        selected.to_csv(os.path.join(outpath, outfile.format(name=name)), index=False)
        print 'done'