"""
Usage:

```shell
$ python process_workspace [path to workspace] [path to output directory]
```
"""

import sys
import os
import pandas as pd
import re

md = sys.argv[1]        # Path to directory containing Hydrus output directories.
outpath = sys.argv[2]   # Path to directory where processed data will be stored.
f_format = '{dataset}_{param}.csv'

# Data for h and theta are stored in separated subdirectories.
h_dir = os.path.join(outpath, 'h')
theta_dir = os.path.join(outpath, 'theta')

# Create the subdirectories for h and theta, if they don't already exist.
if not os.path.exists(h_dir):
    os.makedirs(h_dir)
if not os.path.exists(theta_dir):
    os.makedirs(theta_dir)

# Ignore any file or directory that does not contain an 'Obs_Node.out' file.
obs_path = os.path.join(md, 'Obs_Node.out')
if os.path.exists(obs_path):
    head, d = os.path.split(md)
    print d,"::",

    # Output files are named by the Hydrus output directory name and parameter.
    h_path = os.path.join(h_dir, f_format.format(dataset=d, param='h'))
    theta_path = os.path.join(theta_dir, f_format.format(dataset=d, param='theta'))

    # Skip any directories for which an h data file already exists.
    if not os.path.exists(h_path):
        print 'processing...',
        with open(obs_path, 'r') as f:
            raw = f.readlines()

        data = []
        for line in raw[10:-1]:    # Discard Hydrus headers and "end" line.
            # The Hydrus data file is white-space delimited.
            data.append(re.split('\s+', line))

        columns = data[0][0:2] + data[0][2:5]*61 + [data[0][-1]]
        df = pd.DataFrame(data[1:], columns=columns)
        df.h.to_csv(h_path, header=False, index=False)
        df.theta.to_csv(theta_path, header=False, index=False)
        print 'done'
    else:
        print 'already processed or name conflict, skipping'
