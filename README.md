# HydrusHelper
Python CLI tool for streamlining Hydrus simulations

## Usage
The python runner enters the root fraction and climate data into the hydrus
program based on csv files ``beta.csv`` and ``atmosph.csv``. An example hydrus
workspace (with the correct soil profile/time/parmeter info) is needed for the
code to replicate for each sequential simulation. The simulation names are must
be entered one per line and saved in the ``simulation_names.dat`` file. The time
points for each simulation is saved one per line in ``simulations.dat``.
The file will run the soil profile with 101 nodes which it populates in the
nodes section of the ``profile.dat`` file.

See [testdata/input](testdata/input) for an example configuration.

The inputs for python runner are as follows:
``-n`` (name)
``-i`` file path for input
``-s`` file path for simulations
``-o`` file path for output
``--chain-theta``

When runner is used the code will generate the first simulation file and ask the
user to run the hydrus file from a separate command prompt and hit enter upon a
complete run of the hydrus file. Upon hitting enter, the last time point’s
theta value for the previous simulation will be read in as an initial value for
the next simulation. This will be repeated for the number of simulations in the
simulation files.

``process_workspace.py`` will pull out ``h``, ``theta``, and ``ET`` for all
files. The inputs needed are:

* Path to directory containing Hydrus output
* Path to directory where processed data will be stored.

## Getting help

* Raise an [issue](https://github.com/erickpeirson/HydrusHelper/issues), or...
* Contact Heather Kropp:

Postdoctoral Researcher
Department of Geography
Colgate University
315.228.7027
hkropp@colgate.edu

## Tests

Unit tests can be found in ``tests.py``.
