import numpy as np
import pandas as pd
import os
import re
import shutil
import copy
import argparse


# Convert an array of floats to an array of strings, formatted in scientific
#  notation with three digits in the exponent.
scientific = np.vectorize(lambda f: re.sub('(e[+-])', lambda m: m.group(1) + '0', format(f, '0.6e')))


def number(value):
    """
    Gently attempt to coerce an object to ``int`` or ``float``.
    """
    try:
        return int(value)
    except (ValueError, TypeError) as E:
        try:
            return float(value)
        except (ValueError, TypeError) as E:
            return value


def strings(list):
    """
    Coerce objects in a list to ``str``.
    """
    return [str(o) for o in list]


class WPathMixin(object):
    """
    Provides base path behavior to classes that read to/write from disk.
    """
    def __init__(self, wpath, **kwargs):
        """
        ``kwargs`` are blindly assigned as instance attributes.
        """
        self.wpath = wpath
        for k,v in kwargs.iteritems():
            setattr(self, k, v)

    def _path(self, fpath):
        return os.path.join(self.wpath, fpath)

    def _checkpath(self, fpath):
        spath = self._path(fpath)
        if not os.path.exists(spath):
            raise RuntimeError("Can't find %s in %s." % (fpath, self.wpath))
        return spath


class TemplateMixin(object):
    """
    Provides template loading behavior.
    """

    def get_template(self):
        """
        Retrieves the unformatted body of the template.
        """
        template = getattr(self, 'template', None)
        if template is None:
            raise AttributeError('TemplateMixin children must have `template`')

        base_path, _ = os.path.split(os.path.realpath(__file__))
        template_path = os.path.join(base_path, 'templates', template)

        if not os.path.exists(template_path):
            raise RuntimeError('Could not find template %s' % template)

        with open(template_path, 'r') as f:
            template_body = f.read()
        return template_body


    def render_template(self, **data):
        """
        Fill the template with keyword parameters in ``data``.
        """
        return self.get_template().format(**data)


class DataReader(WPathMixin):
    """
    Provides file-reading behavior.
    """
    def _get_raw(self):
        if not os.path.exists(self.wpath):
            raise RuntimeError('Specified path does not contain data')

        with open(self.wpath, 'r') as f:
            raw = f.readlines()
        return raw


class DataWriter(object):
    """
    Provides file-writing behavior.
    """
    def get_param(self, param):
        return getattr(self, param)

    def get_params(self, params):
        combined = np.array([self.get_param(param) for param in params])
        return combined.reshape((combined.shape[0], combined.shape[1])).T

    def write_params(self, params, path):
        df = pd.DataFrame(self.get_params(params), columns=params)
        df.to_csv(path, index=False)

    def write_param(self, param, path):
        df = pd.DataFrame(self.get_param(param))
        df.to_csv(path, index=False, header=False)


class Cloneable(object):
    """
    Provides memory and disk cloneing to subclasses.
    """
    def clone_to(self, tpath):
        cloned = copy.deepcopy(self)
        cloned.wpath = tpath
        return cloned

    def write(self):
        if not hasattr(self, 'render'):
            raise AttributeError('Cloneable subclass must provide `render()`')

        with open(self.wpath, 'w') as f:
            f.write(self.render())


class Hydrus1DDATHandler(DataReader):
    def read(self):
        raw = self._get_raw()

        main = {}
        profile = {}
        for line in [ line.strip().split('=') for line in raw ]:
            if line[0] == '[Main]': use = main
            elif line[0] == '[Profile]': use = profile

            if len(line) > 1:
                use[line[0]] = number(line[1])
        setattr(self, 'main', main)
        setattr(self, 'profile', profile)

        if 'ProfileDepth' in profile and 'NumberOfNodes' in profile:
            assert profile['ProfileDepth'] == profile['NumberOfNodes'] - 1

    def write(self):
        """
        Generate and write Hydrus1D config file to disk.
        """

        # Sanity check.
        if 'ProfileDepth' in self.profile and 'NumberOfNodes' in self.profile:
            assert self.profile['ProfileDepth'] == self.profile['NumberOfNodes'] - 1

        out = [';', '[Main]']
        for k,v in self.main.iteritems():
            if type(v) is float:    # Reformat exponent notation for Hydrus.
                v = format(v, '0.0E')
            out.append('{0}={1}'.format(k,v))
        out += [';', '[Profile]']

        # Reformat exponent notation for Hydrus.
        for k,v in self.profile.iteritems():
            if type(v) is float:
                v = re.sub('([0-9])E', lambda m: m.group(1) + '.E', format(v, '0.0E'))
            out.append('{0}={1}'.format(k,v))

        with open(self.wpath, 'w') as f:
            f.write('\n'.join(out))


class TLevelHandler(DataReader, DataWriter):
    headers_seen = False
    end_seen = False

    def read(self):
        self.classifier = {
            '*******'   :   lambda x: None,
            'Welcome'   :   lambda x: None,
            'Date:'     :   lambda x: setattr(self, 'date', x),
            'Units:'    :   lambda x: setattr(self, 'units', x),
            ''          :   lambda x: None,
            'Time'      :   lambda x: setattr(self, 'headers', x),
            'end'       :   lambda x: None,
            '[T]'       :   lambda x: None
        }

        if hasattr(self, 'headers'):
            delattr(self, 'headers')

        raw = self._get_raw()

        # Data are whitespace delimited.
        proc = [ re.split('\s+', line.strip()) for line in raw ]

        data = []
        for line in proc:
            try:
                self.classifier[line[0]](line)
            except KeyError:
                if hasattr(self, 'headers'):
                    data.append(line)

        df = pd.DataFrame(data, columns=self.headers)
        params = list(set(set(df.columns.values)))
        for param in params:
            setattr(self, param, df[[param]].astype(float).values)


class SELECTORHandler(DataReader, TemplateMixin):
    template = 'SELECTOR.template'    # Required by TemplateMixin.

    def render(self):
        if not hasattr(self, 'tMax'):
            raise RuntimeError('tMax not specified')

        # TODO: Include other parameters.
        return self.render_template(**self.__dict__)

    def write(self):
        """
        (over)Write the SELECTOR.IN file at self.wpath
        """

        with open(self.wpath, 'w') as f:
            f.write(self.render())

    def read(self):
        """
        """

        def set_params(params, values):
            for p,v in zip(params, values):
                if not hasattr(self, p):        # Don't overwrite values.
                    setattr(self, p, v)

        self.classifier = {
            '*******'   :  lambda x: None,
            'Welcome'   :  lambda x: None,
            'Heading'   :  lambda x: None,
            '***'       :  lambda x: None,
            'Pcp_File_Version=4': lambda x: None,
            'LUnit'     :   lambda x: set_params(['LUnit', 'TUnit', 'MUnit'], [x[c+1][0], x[c+2][0], x[c+3][0]]),
            'lWat'      :   lambda x: set_params(x[c], x[c+1]),
            'lSnow'     :   lambda x: set_params(x[c], x[c+1]),
            'NMat'      :   lambda x: set_params(x[c], x[c+1]),
            'MaxIt'     :   lambda x: set_params(x[c][0:3], x[c+1]),
            'TopInf'    :   lambda x: set_params(x[c], x[c+1]),
            'BotInf'    :   lambda x: set_params(x[c], x[c+1]),
            'hTab1'     :   lambda x: set_params(x[c], x[c+1]),
            'Model'     :   lambda x: set_params(x[c], x[c+1]),
            'thr'       :   lambda x: set_params(x[c], x[c+1]),
            'dt'        :   lambda x: set_params(x[c], x[c+1]),
            'tInit'     :   lambda x: set_params(x[c], x[c+1]),
            'lPrintD'   :   lambda x: set_params(x[c], x[c+1]),
        }

        raw = self._get_raw()

        # Data are whitespace delimited.
        proc = [re.split('\s+', line.strip()) for line in raw]

        c = 0
        for line in proc:
            try:
                self.classifier[line[0]](proc)
            except KeyError:
                pass
            c += 1


class ObsNodeHandler(DataReader, DataWriter):
    def read(self):
        self.classifier = {
            '*******':  lambda x: None,
            'Welcome':  lambda x: None,
            'Date:':    lambda x: setattr(self, 'date', x),
            'Units:':   lambda x: setattr(self, 'units', x),
            '':         lambda x: None,
            'time':     lambda x: setattr(self, 'headers', x),
            'end':      lambda x: None,
            '[T]':      lambda x: None,
            'Node(':    lambda x: None
        }

        if hasattr(self, 'headers'):
            delattr(self, 'headers')

        raw = self._get_raw()

        # Data are whitespace delimited.
        proc = [ re.split('\s+', line.strip()) for line in raw ]

        data = []
        for line in proc:
            try:
                self.classifier[line[0]](line)
            except KeyError:
                if hasattr(self, 'headers'):
                    data.append(line)
        # Infer the number of nodes by the shape of the data.
        self.N = int((len(data[0])-1.)/3.)

        # Generate headers for the data. If Hydrus was run without configuring the node
        #  number in HYDRUS1D.DAT, then there may not be a sufficient number of headers
        #  for the data.
        columns = [self.headers[0]] + self.headers[1:4]*self.N
        df = pd.DataFrame(data, columns=columns)

        self.h = df.h.astype(float).values
        self.theta = df.theta.astype(float).values
        self.Flux = df.theta.astype(float).values


class ATMOSPHHandler(DataReader, Cloneable, TemplateMixin):
    """
    ATMOSPH.IN
    """
    delim = '   '
    template = 'ATMOSPH.template'

    defaults = {
        'DailyVar'  : 'f',
        'SinusVar'  : 'f',
        'lLay'      : 'f',
        'lBCCycles' : 'f',
        'lInterc'   : 'f',
        'lDummy1'   : 'f',
        'lDummy2'   : 'f',
        'lDummy3'   : 'f',
        'lDummy4'   : 'f',
        'lDummy5'   : 'f',
        'hCritS'    : 0,
    }

    dtypes = {
        'MaxAL'     :   int,
        'hCritS'    :   int,
        'tAtm'      :   int,
        'Prec'      :   float,
        'rSoil'     :   float,
        'rRoot'     :   float,
        'hCritA'    :   int,
        'rB'        :   int,
        'hB'        :   int,
        'ht'        :   int,
        'DailyVar'  :   str,
        'SinusVar'  :   str,
        'lLay'      :   str,
        'lBCCycles' :   str,
        'lInterc'   :   str,
        'lDummy1'   :   str,
        'lDummy2'   :   str,
        'lDummy3'   :   str,
        'lDummy4'   :   str,
        'lDummy5'   :   str,
    }

    def read(self):
        self.classifier = {
            '*******'   :   lambda x: None,
            '***'       :   lambda x: None,
            'Pcp_File_Version=4': lambda x: None,
            'Welcome'   :   lambda x: None,
            ''          :   lambda x: None,
            'MaxAL'     :   lambda x: setattr(self, 'MaxAL', int(x[c+1][0])),
            'DailyVar'  :   lambda x: [ setattr(self, x[c][i], x[c+1][i]) for i in xrange(len(x[c+1])) if x[c][i] != 'lDummy' ],
            'hCritS'    :   lambda x: setattr(self, 'hCritS', int(x[c+1][0])),
            'end'       :   lambda x: None,
            'end***'    :   lambda x: None,  # May have extra headers.
            'tAtm'      :   lambda x: setattr(self, 'headers', x[c][:len(x[c+1])]),
        }

        if hasattr(self, 'headers'):
            delattr(self, 'headers')

        raw = self._get_raw()

        # Data are whitespace delimited.
        proc = [ re.split('\s+', line.strip()) for line in raw ]

        data = []
        c = 0
        for line in proc:
            try:
                self.classifier[line[0]](proc)
            except KeyError:
                if hasattr(self, 'headers'):
                    data.append(line)
            c += 1

        df = pd.DataFrame(data, columns=self.headers)
        params = list(set(set(df.columns.values)))
        for param in params:
            vals = df[[param]].astype(self.dtypes[param]).values

            # Each parameter should have precisely MaxAL values.
            if hasattr(self, 'MaxAL'):
                assert vals.shape[0] == self.MaxAL
            setattr(self, param, vals.reshape(vals.shape[0]))

        # Populate missing parameters with default values.
        for key, value in self.defaults.iteritems():
            if not hasattr(self, key):
                setattr(self, key, value)

    def _generate_data(self):

        if not hasattr(self, 'tAtm'): setattr(self, 'tAtm', np.arange(1, self.MaxAL+1, dtype='int32'))
        if not hasattr(self, 'rB'): setattr(self, 'rB', np.zeros((self.MaxAL,), dtype='int32'))
        if not hasattr(self, 'hB'): setattr(self, 'hB', np.zeros((self.MaxAL,), dtype='int32'))
        if not hasattr(self, 'ht'): setattr(self, 'ht', np.zeros((self.MaxAL,), dtype='int32'))
        if not hasattr(self, 'hCritA'): setattr(self, 'hCritA', np.array([ 100000 ] *self.MaxAL))

        # Check for data completeness.
        for attr in self.dtypes.keys():
            if not hasattr(self, attr):
                raise RuntimeError('Insufficient data, missing {0}'.format(attr))

        # tAtm must always start at 1
        assert self.tAtm.min() == 1
        assert self.tAtm.max() == self.MaxAL

        # Pull all data together, and format for insertion in the template.
        compiled = zip(self.tAtm,
                        self.Prec,
                        self.rSoil,
                        self.rRoot,
                        self.hCritA,
                        self.rB,
                        self.hB,
                        self.ht)

        self.data = '\n'.join([self.delim + self.delim.join(strings(compiled[i]))
                                  for i in xrange(len(compiled)) ])

    def render(self):
        """
        Generate the full string representation of PROFILE.DAT.

        Returns
        -------
        str
            Fully rendered content for PROFILE.DAT
        """

        self._generate_data()

        fdict = { k:getattr(self, k) for k in self.dtypes.keys() }
        headers = self.delim + self.delim.join(self.headers + ['RootDepth'])
        fdict.update({'headers': headers, 'data':self.data})

        try:
            formatted = self.render_template(**fdict)
        except AttributeError:
            raise RuntimeError('Required parameters not set.')

        return formatted


class ProfileHandler(DataReader, Cloneable, TemplateMixin):
    delim = '   '
    template = 'Profile.template'
    headers = ['node', 'x', 'h', 'Mat', 'Lay', 'Beta', 'Axz', 'Bxz', 'Dxz']

    p0 = 0
    p1 = 0
    p2 = 1

    def _generate_data(self):
        for attr in ['N', 'Beta','h']:
            if not hasattr(self, attr):
                raise RuntimeError('Insufficient data')

        # Calculate node numbers and depths.
        self.node = np.arange(1,self.N+1)         # Node numbers 1 through N.
        self.x = (self.node-1.)*-1.             # Node depths.
        self.x_sfc = self.x.max()   # Surface node depth.
        self.x_dpt = self.x.min()               # Deepest node depth.

        # TODO: Make these configurable, to support multiple materials and layers.
        self.Mat = np.ones((self.N,), dtype='int32')    # Material.
        self.Lay = np.ones((self.N,), dtype='int32')    # Layer.

        # These are always 1. Setting these != 1 would involve modifications to other
        #  files, and is beyond the scope of current needs.
        self.Axz = np.ones((self.N,))
        self.Bxz = np.ones((self.N,))
        self.Dxz = np.ones((self.N,))

        # Pull all data together, and format for insertion in the template.
        compiled = zip(self.node,
                        scientific(self.x),
                        scientific(self.h),
                        self.Mat, self.Lay,
                        scientific(self.Beta),
                        scientific(self.Axz),
                        scientific(self.Bxz),
                        scientific(self.Dxz))
        self.data = '\n'.join(
            [ self.delim + self.delim.join(strings(compiled[i]))
                for i in xrange(len(compiled)) ])

    def render(self):
        """
        Generate the full string representation of PROFILE.DAT.

        Returns
        -------
        str
            Fully rendered content for PROFILE.DAT
        """

        self._generate_data()
        headers = self.delim.join(self.headers[1:] + ['Temp','Conc','SConc'])
        node_numbers = self.delim.join([str(i) for i in range(1, self.N+1)])

        try:
            formatted = self.render_template(
                    x_sfc=scientific(self.x_sfc),
                    x_dpt=scientific(self.x_dpt),
                    nodes=self.N,
                    p0=self.p0,
                    p1=self.p1,
                    p2=self.p2,             # No header for node number.
                    headers=headers,
                    data=self.data,
                    node_numbers=node_numbers)
        except AttributeError:
            raise RuntimeError('Required parameters not set.')

        return formatted

    def write(self):
        """
        (over)Write the PROFILE.DAT file at self.wpath
        """

        with open(self.wpath, 'w') as f:
            f.write(self.render())

    def read(self):
        """
        Load data from an existing PROFILE.DAT file.
        """

        raw = self._get_raw()
        proc = [ re.split('\s+', line.strip()) for line in raw ]

        if proc[0][0][0:3] != 'Pcp':
            raise RuntimeError(
                'Unexpected value in header line; file may have been tampered with.')
        try:
            expect = int(proc[1][0])
        except ValueError:
            raise RuntimeError(
                'Unexpected value on line 1; file may have been tampered with.')

        self.x_sfc = float(proc[2][1])
        self.x_dpt = float(proc[3][1])

        if not hasattr(self, 'N'):
            self.N = int(proc[2+expect][0])   # This should be the number of nodes.

        self.p0 = int(proc[2+expect][1])
        self.p1 = int(proc[2+expect][2])
        self.p2 = int(proc[2+expect][3])
        data = proc[3+expect:3+expect+self.N]

        df = pd.DataFrame(data, columns=self.headers)

        for attr in self.headers:
            vals = df[[attr]].astype(float).values
            vals = vals.reshape(vals.shape[0])
            setattr(self, attr, vals)

        # Double-check that the correct number of rows was read.
        assert self.N == self.h.shape[0]
        assert self.N == self.node.max()

class WorkspaceHandler(WPathMixin):
    handlers_classes = {
        'Obs_Node.out'  :   ObsNodeHandler,
        'PROFILE.DAT'   :   ProfileHandler,
        'T_Level.out'   :   TLevelHandler,
        'HYDRUS1D.DAT'  :   Hydrus1DDATHandler,
        'ATMOSPH.IN'    :   ATMOSPHHandler,
        'SELECTOR.IN'   :   SELECTORHandler,
    }

    def __init__(self, wpath, **kwargs):
        super(WorkspaceHandler, self).__init__(wpath, **kwargs)
                                # vvv Skip any "hidden" files (i.e. starting with `.`).
        self.fnames =  [ fname for fname in os.listdir(self.wpath) if fname[0] != '.' ]
        self.handlers = {}

        if not hasattr(self, 'name'):
            raise ValueError('Please provide a name for this workspace.')


    def load_handlers(self):
        """
        Inspect files in ``self.wpath``, and load handlers if available.
        """
        self.fnames =  [ fname for fname in os.listdir(self.wpath) if fname[0] != '.' ]
        for fname in self.fnames:
            if fname in self.handlers_classes:
                self.handlers[fname] = self.handlers_classes[fname](os.path.join(self.wpath, fname))
                # TODO: Not everything should be read.
                self.handlers[fname].read()

    def clone_into(self, tpath, name):
        """
        Copy all of the files in the workspace into a new workspace at ``tpath``.

        Parameters
        ----------
        tpath : str
            Path to a new Hydrus workspace (directory). If no directory exists at this
            path, then a new one will be created.

        Returns
        -------
        :class:`.WorkspaceHandler`
            A new handler, with ``wpath == tpath``.
        """

        try:     # Create a new directory at tpath, if needed.
            if not os.path.exists(tpath):
                os.makedirs(tpath)
        except IOError:
            raise ValueError(''.join(
                ['Cannot write to {0}. Make sure that the parent directory exists, and',
                 ' that python has permission to write to it.']))

        # Copy files with MOST metadata. See https://docs.python.org/2/library/shutil.html
        #  for relevant warnings.
        for fname in self.fnames:
            shutil.copy(os.path.join(self.wpath, fname), os.path.join(tpath, fname))

        return WorkspaceHandler(tpath, name=name)

class SimulationHandler(WPathMixin):
    """
    Singleton class for orchestrating the simulation series.
    """

    chain_theta = False
    simulation_workspaces = {}
    simulation_paths = {}

    def __init__(self, wpath, **kwargs):
        super(SimulationHandler, self).__init__(wpath, **kwargs)

        self.read_simulations()
        self.read_simulation_names()

        self.read_beta()

        if not self.chain_theta:
            self.read_theta()

        self.read_atmosph()
        self.read_workspace()

        if not hasattr(self, 'name'):
            raise ValueError('Please provide a name for this simulation set.')
        if not hasattr(self, 'simpath'):
            raise ValueError('Please provide a path to save workspaces.')
        if not hasattr(self, 'outpath'):
            raise ValueError('Please provide a path to an output directory.')

        self.current = 0

    def next(self):
        """
        Create and populate the next workspace with data.
        """

        if self.current >= len(self.simulations):
            # No further simulations to run.
            return False

        # Clone the workspace.
        self.generate_workspace(self.current)

        # PROFILE.DAT
        phandler = self.simulation_workspaces[self.current].handlers['PROFILE.DAT']
        phandler.read()
        phandler.N = self.N
        if hasattr(self, 'beta'):
            phandler.Beta = self.beta[:,self.current]
        if not self.chain_theta and hasattr(self, 'theta'):
            phandler.h = self.theta

        if self.chain_theta and self.current > 0:
            phandler.h = self.simulation_workspaces[self.current - 1].handlers['Obs_Node.out'].theta[-1,:]
        phandler.write()

        # ATMOSPH.IN
        ahandler = self.simulation_workspaces[self.current].handlers['ATMOSPH.IN']
        ahandler.read()

        start = sum(self.simulations[:self.current]) + 1
        end = start + self.simulations[self.current]
        duration = end - start
        present = set()
        for attr in ['Prec', 'rSoil', 'rRoot', 'hCritA']:
            if hasattr(self, attr):
                present.add(attr)

                setattr(ahandler, attr, getattr(self, attr)[start-1:end-1])
        for param in list(set(['rB', 'hB', 'ht', 'hCritA']) - present):
            delattr(ahandler, param)

        setattr(ahandler, 'tAtm', np.arange(1, 1 + duration))
        ahandler.MaxAL = duration
        ahandler.write()

        # HYDRUS1D.DAT

        hhandler = self.simulation_workspaces[self.current].handlers['HYDRUS1D.DAT']
        hhandler.read()

        hhandler.profile['NumberOfNodes'] = self.N
        hhandler.write()

        # SELECTOR.IN
        shandler = self.simulation_workspaces[self.current].handlers['SELECTOR.IN']
        shandler.read()
        shandler.tMax = duration
        shandler.write()

        return True

    def post_run(self):
        self.simulation_workspaces[self.current].load_handlers()
        obs_handler = self.simulation_workspaces[self.current].handlers['Obs_Node.out']
        obs_handler.read()

        tlevel_handler = self.simulation_workspaces[self.current].handlers['T_Level.out']
        tlevel_handler.read()

        self.current += 1

    def read_workspace(self):
        """
        Expects to find a workspace called `workspace` in `self.wpath`. This
        workspace will be the template for all workspaces in this set of simulations.
        """
        self.template_workspace = WorkspaceHandler(self._checkpath('workspace'), name='template')

    def generate_workspace(self, simulation):
        """
        Set up a new (cloned) workspace with data for simulation `simulation`.

        Parameters
        ----------
        simulation : int
            0-based index into ``self.simulations``.
        """

        if simulation >= len(self.simulations):
            raise ValueError(
                'The selected simulation exceeds the number of known simulations')

        if hasattr(self, 'simulation_names'):
            sname = self.simulation_names[simulation]
        else:
            sname = str('sim'+simulation)

        spath = os.path.join(self.simpath, ('{0}_{1}'.format(self.name, sname)))
        self.simulation_paths[simulation] = spath
        self.simulation_workspaces[simulation] = self.template_workspace.clone_into(spath, sname)
        self.simulation_workspaces[simulation].load_handlers()

    def read_simulations(self):
        """
        Expects to find a file called ``simulations.dat`` in ``self.wpath``.

        Each line corresponds to a simulation, and should contain a single integer
        specifying the number of rows in the data that should be used in that simulation.
        """

        spath = self._checkpath('simulations.dat')

        with open(spath, 'rU') as f:
            self.simulations = [ int(l.strip()) for l in f.readlines() ]

        # There should be one column of beta for each simulation.
        if hasattr(self, 'beta'):
            assert len(self.simulations) == self.beta.shape[1]

    def read_simulation_names(self):
        """
        Expects to find a file called ``simulation_names.dat`` in ``self.wpath``.

        Each line corresponds to a simulation, and should contain a string that will be
        used to generate workspace names. If no such file exists, will exit quietly.
        The number of lines in ``simulation_names.dat`` should be equal to the number of
        lines in ``simulations.dat``, otherwise an AssertionError is thrown.
        """

        try:
            spath = self._checkpath('simulation_names.dat')
        except RuntimeError:
            return

        with open(spath, 'rU') as f:
            self.simulation_names = [ l.strip() for l in f.readlines() ]

        # One simulation name per simulation.
        if hasattr(self, 'simulations'):
            assert len(self.simulations) == len(self.simulation_names)

    def read_csv(self, param, fname):
        spath = self._checkpath(fname)
        df = pd.read_csv(spath, header=None)        # No header row.
        setattr(self, param, df.astype(float).values)         # Save only the data array.

    def read_beta(self):
        """
        Expects to find a file called ``beta.csv`` in ``self.wpath``.

        Each column corresponds to one simulation. Thus the number of columns of beta
        should equal the number of lines in ``simulations.dat``.
        """

        self.read_csv('beta', 'beta.csv')

        # There should be one column of beta for each simulation.
        if hasattr(self, 'simulations'):
            assert len(self.simulations) == self.beta.shape[1]

        # If the number of nodes (N) hasn't been set explicitly, infer it from the data.
        if not hasattr(self, 'N'):
            self.N = self.beta.shape[0]

    def read_theta(self):
        """
        Expects to find a file called ``theta.csv`` in ``self.wpath``.

        Each column corresponds to one simulation. Thus the number of columns of theta
        should equal the number of lines in ``simulations.dat``.
        """

        self.read_csv('theta', 'theta.csv')

        # There should be one column of theta for each simulation.
        if hasattr(self, 'simulations'):
            assert len(self.simulations) == self.beta.shape[1]

    def read_atmosph(self):
        """
        Expects to find a file called ``atmosph.csv`` in ``self.wpath``.

        Each column (with header) provides data for a parameter in ATMOSPH.IN. Data for
        all simulations should be provided in sequential rows. For example, if simulations
        are for sequential periods over the course of a year (365 days), then there should
        be 365 rows of data. More precisely, there should be at least as many rows as the
        sum of the values in `simulations.dat`.
        """

        spath = self._checkpath('atmosph.csv')

        df = pd.read_csv(spath)

        for attr in df.columns:
            setattr(self, attr, getattr(df, attr).values)

        if hasattr(self, 'simulations'):
            try:
                assert self.tAtm.shape[0] >= sum(self.simulations)
            except AssertionError:
                raise ValueError(''.join(
                    ['Data file `atmosph.csv` contains too few rows for the simulations',
                     ' specified in `simulations.dat`. The number of rows should be',
                     ' equal to or greater than the sum of the values in',
                     ' `simulations.dat`.']))

        if not hasattr(self, 'hCritA'): setattr(self, 'hCritA', np.array([100000]*self.tAtm.shape[0]))

    def process_output(self):
        """
        Extract output from Obs_Node.out and T_Level.out, and write to CSV.
        """

        # Create a directory at outpath, if it doesn't exist.
        try:
            if not os.path.exists(self.outpath):
                os.makedirs(self.outpath)
        except IOError:
            raise RuntimeError('Could not create directory at {0}'.format(outpath))


        for sim, whandler in self.simulation_workspaces.iteritems():
            obs_handler = whandler.handlers['Obs_Node.out']
            tlevel_handler = whandler.handlers['T_Level.out']

            # h
            h_ = obs_handler.get_param('h')
            if not hasattr(self, 'h'):
                self.h = h_
            else:
                self.h = np.append(self.h, h_, axis=0)

            # theta
            theta_ = obs_handler.get_param('theta')
            if not hasattr(self, 'theta'):
                self.theta = theta_
            else:
                self.theta = np.append(self.theta, theta_, axis=0)

            # tlevel params
            params = ['rRoot', 'vRoot', 'rTop', 'vTop', 'sum(Evap)']
            tlevel_ = tlevel_handler.get_params(params)
            if not hasattr(self, 'tlevel'):
                self.tlevel = tlevel_
            else:
                self.tlevel = np.append(self.tlevel, tlevel_, axis=0)

        hpath = os.path.join(self.outpath, '{0}_h.csv'.format(self.name))
        tpath = os.path.join(self.outpath, '{0}_theta.csv'.format(self.name))
        opath = os.path.join(self.outpath, '{0}_tlevel.csv'.format(self.name))

        df_h = pd.DataFrame(self.h)
        df_h.to_csv(hpath, index=False, header=False)

        df_t = pd.DataFrame(self.theta)
        df_t.to_csv(tpath, index=False, header=False)

        df_o = pd.DataFrame(self.tlevel, columns=params)
        df_o.to_csv(opath, index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulation helper for HYDRUS1D')
    parser.add_argument('-n', metavar='name', required=True,
                        help='Provide a name for this simulation set')
    parser.add_argument('-i', metavar='input-path', required=True,
                        help='Path to input directory')
    parser.add_argument('-s', metavar='sim-path', required=True,
                        help='Directory where workspaces will be created')
    parser.add_argument('-o', metavar='output-path', required=True,
                        help='Path to directory where output will be stored')
    parser.add_argument('--chain-theta', action='store_const', const=True,
                        default=False,
                        help=' '.join(['If provided, final theta values from',
                                       'Obs_Node.out in one simulation will',
                                       'be used as values of h in PROFILE.DAT',
                                       'in the next simulation']))

    args = parser.parse_args()
    handler = SimulationHandler(args.i, simpath=args.s, outpath=args.o, name=args.n, chain_theta=args.chain_theta)

    handler.read_simulations()

    while handler.next():
        raw_input('Run HYDRUS on the workspace {0}, then press ENTER'.format(handler.simulation_names[handler.current]))
        handler.post_run()

    handler.process_output()
