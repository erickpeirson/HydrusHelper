# Usage:
#  python change_nodes.py [path to workspace]

import sys
import os
import subprocess
import re

hpath = os.path.join(sys.argv[1], 'PROFILE.DAT')

if not os.path.exists(hpath):
    raise ValueError("Invalid path!")

with open(hpath, 'rU') as f:
    print "reading", hpath
    profile = [ l.strip('\n') for l in f.readlines() ]

profile[-1] = '   '.join(['']+ [str(i) for i in range(1,62)])
profile[-2] = '   61'
    
with open(hpath, 'w') as f:
    print "writing", hpath, "...",
    f.write('\n'.join(profile))
    print "done."
