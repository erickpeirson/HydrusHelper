import unittest
import tempfile
import numpy as np
import pandas as pd
import os
from runner import *

class TestDataWriter(unittest.TestCase):
    def test_write(self):
        handler = DataWriter()
        setattr(handler, 'bob', np.random.random((5,)))
        
        i,tpath = tempfile.mkstemp()
        handler.write_param('bob', tpath)
        
        df = pd.read_csv(tpath, header=None)
        self.assertEqual(df[[0]].shape[0], 5)

class TestSimulationHandler(unittest.TestCase):
    def setUp(self):
        self.wpath = './testdata/input'
        self.simpath = './testdata/simulations'
        self.outpath = './testdata/simout'
        self.handler = SimulationHandler(
                            self.wpath, simpath=self.simpath, outpath=self.outpath, 
                            name='test', chain_theta=True)
    
    def test_read_simulations(self):
        """
        The test dataset has five simulations.
        """
        
        self.handler.read_simulations()
        self.assertEqual(len(self.handler.simulations), 6)
    
    def test_read_beta(self):
        """
        There should be one column of beta for each simulation.
        """
        
        self.handler.read_simulations()    
        self.handler.read_beta()
        self.assertEqual(self.handler.beta.shape[0], 61)
        self.assertEqual(self.handler.beta.shape[1], 6)
        
        # Should infer N (number of nodes) from beta.
        self.assertEqual(self.handler.N, 61)
        
    def test_read_atmosph(self):
        self.handler.read_simulations()
        self.handler.read_atmosph()
        
        self.assertGreaterEqual(self.handler.tAtm.shape[0], sum(self.handler.simulations))
        
    def test_read_workspace(self):

        self.handler.read_workspace()
        
        self.assertIsInstance(self.handler.template_workspace, WorkspaceHandler)
        
    def test_generate_workspace(self):
        self.handler.read_workspace()
        self.handler.generate_workspace(0)
        
        self.assertTrue(os.path.exists(os.path.join(self.simpath, 'test_winter')))
        
    def test_next(self):
        self.handler.read_workspace()    
        self.handler.next()

        self.assertTrue(os.path.exists(os.path.join(self.simpath, 'test_winter')))
        
        self.assertEqual(self.handler.beta[0,0], self.handler.simulation_workspaces[0].handlers['PROFILE.DAT'].Beta[0])
        
                
    def test_post_run(self):
        self.handler.read_workspace()
        self.handler.next()
        
        self.handler.post_run()
        
        self.assertEqual(self.handler.current, 1)
        
        self.handler.next()
        
    

class TestWorkspaceHandler(unittest.TestCase):
    def setUp(self):
        self.wpath = './testdata/PRVE_swc_spring1_d6'
        self.handler = WorkspaceHandler(self.wpath, name='test')
    
    def test_load_handlers(self):
        self.handler.load_handlers()
        
        self.assertGreater(len(self.handler.handlers), 0) # some handlers loaded
        for k,handler in self.handler.handlers.iteritems():
            self.assertIn(DataReader, handler.__class__.__bases__)
            
    def test_clone_into(self):
        tpath = tempfile.mkdtemp()
        new_handler = self.handler.clone_into(tpath, name='test_cloned')
        
        # should return a new WorkspaceHandler
        self.assertIsInstance(new_handler, WorkspaceHandler)
        
        # filesizes should be identical
        for fname in self.handler.fnames:
            self.assertEqual(
                os.path.getsize(os.path.join(self.wpath, fname)),
                os.path.getsize(os.path.join(tpath, fname)))

        
class TestATMOSPHHandler(unittest.TestCase):
    def setUp(self):      
        wpath = './testdata/PRVE_swc_spring1_d6/ATMOSPH.IN'        
        self.handler = ATMOSPHHandler(wpath)
    
    def test_read(self):
        self.handler.read()
        self.assertEqual(self.handler.MaxAL, 30)
        self.assertTrue(hasattr(self.handler, 'DailyVar'))
        self.assertTrue(hasattr(self.handler, 'hCritS'))
        self.assertTrue(hasattr(self.handler, 'tAtm'))
        self.assertTrue(hasattr(self.handler, 'rB'))     
        
        self.assertIsInstance(self.handler.rB[0], int)
        self.assertIsInstance(self.handler.rSoil[0], float)         
        
    def test_render(self):
        self.handler.read()
        rendered = self.handler.render()  
        
        self.assertIsInstance(rendered, str)
        
    def test_clone_to(self):
        self.handler.read()
        i,tpath = tempfile.mkstemp()
        cloned = self.handler.clone_to(tpath)
        
        # All attributes should be identical (except wpath).
        for attr in dir(self.handler):
            if attr[0] != '_' and attr != 'wpath' and attr != 'data':
                self.assertTrue(hasattr(cloned, attr))  # TODO: get nosier.

        
class TestCloneable(unittest.TestCase):
    def test_no_render(self):
        """
        If a subclass of :class:`.Cloneable` has no `render` method, should raise 
        an AttributeError.
        """
        class Bob(Cloneable):
            a = 0
            b = 1        
            def __init__(self, wpath):
                self.wpath = wpath
                
        i,wpath = tempfile.mkstemp()            
        inst = Bob(wpath)
        i,tpath = tempfile.mkstemp()
        
        with self.assertRaises(AttributeError):
            inst.clone_to(tpath).write()
        
    def test_render(self):
        class RBob(Cloneable):
            a = 0
            b = 1        
            
            def __init__(self, wpath):
                self.wpath = wpath        
                
            def render(self):
                return 'wtf'
        
        i,wpath = tempfile.mkstemp()            
        inst = RBob(wpath)
        inst.write()
        
        i,tpath = tempfile.mkstemp()
        cloned = inst.clone_to(tpath)
        cloned.write()
        
        # Should return another instance of the same class.
        self.assertIsInstance(cloned, RBob)
        
        # All attributes should be identical (except wpath).
        for attr in dir(inst):
            val = getattr(inst, attr)
            if attr[0] != '_' and attr != 'wpath' and type(val).__name__ not in  ['function', 'instancemethod']:
                self.assertEqual(val, getattr(cloned, attr))

        

class TestDataReader(unittest.TestCase):
    def setUp(self):
        wpath = './testdata/PRVE_swc_spring1_d6/Obs_Node.out'        
        self.handler = DataReader(wpath, asdf=1234)
    
    def test_init(self):
        self.assertEqual(self.handler.asdf, 1234)
    
    def test_badpath(self):
        handler = DataReader('/bad/path')
        with self.assertRaises(RuntimeError):
            handler._get_raw()
            
    def test_goodpath(self):
        raw = self.handler._get_raw()
        self.assertIsInstance(raw, list)

class TestTLevelHandler(unittest.TestCase):
    def setUp(self):
        self.wpath = './testdata/PRVE_swc_spring1_d6/T_Level.out'
        self.handler = TLevelHandler(self.wpath)
    
    def test_read(self):
        self.handler.read()
        self.assertTrue(hasattr(self.handler, 'Time'))
        self.assertTrue(hasattr(self.handler, 'sum(rTop)'))

        
        self.assertEqual(self.handler.Time.shape[0], 30)
        self.assertEqual(self.handler.Time.shape[1], 1)              
        

class TestObsNodeHandler(unittest.TestCase):
    def setUp(self):
        self.wpath = './testdata/PRVE_swc_spring1_d6/Obs_Node.out'
        self.handler = ObsNodeHandler(self.wpath)
    
    def test_read(self):
        self.handler.read()
        self.assertEqual(self.handler.N, 61)
        self.assertEqual(self.handler.h.shape[1], self.handler.N)
        self.assertEqual(self.handler.theta.shape[1], self.handler.N)        
        self.assertEqual(self.handler.Flux.shape[1], self.handler.N)

class TestProfileHandler(unittest.TestCase):
    def setUp(self):
        i,self.wpath = tempfile.mkstemp()
        N = 61
        Beta = np.round(np.random.random((N,)), 6)
        h = np.round(np.random.random((N,)), 6)
        self.handler = ProfileHandler(self.wpath, N=N, h=h, Beta=Beta)
    
    def test_write(self):
        self.handler.write()
        
        handler2 = ProfileHandler(self.wpath)
        handler2.read()
        self.assertEqual(self.handler.h.shape[0], handler2.h.shape[0])
        self.assertSequenceEqual(list(self.handler.h), list(handler2.h))
        
        self.assertEqual(self.handler.N, self.handler.h.shape[0])
        self.assertEqual(handler2.N, handler2.h.shape[0])        
        
    def test_write_nodata(self):
        i,self.wpath = tempfile.mkstemp()    
        handler = ProfileHandler(self.wpath)
        with self.assertRaises(RuntimeError):
            handler.write()
            
    def test_read(self):
        wpath = './testdata/PRVE_swc_spring1_d6/PROFILE.DAT'
        handler = ProfileHandler(wpath)
        handler.read()
        self.assertTrue(hasattr(self.handler, 'h'))
        
    def test_read_write(self):
        wpath = './testdata/PRVE_swc_spring1_d6/PROFILE.DAT'
        handler = ProfileHandler(wpath)
        handler.read()
        
        i,wpath2 = tempfile.mkstemp()
        handler = ProfileHandler(wpath2, N=handler.N, Beta=handler.Beta, h=handler.h)
        handler.write()
            
class TestHydrus1DDatHandler(unittest.TestCase):
    def test_read(self):
        wpath = './testdata/PRVE_swc_spring1_d6/HYDRUS1D.DAT'
        handler = Hydrus1DDATHandler(wpath)
        handler.read()
        
        self.assertTrue(hasattr(handler, 'main'))
        self.assertTrue(hasattr(handler, 'profile'))
        self.assertGreater(len(handler.main), 0)
        self.assertGreater(len(handler.profile), 0)        
    
    def test_write(self):

        wpath = './testdata/PRVE_swc_spring1_d6/HYDRUS1D.DAT'
        handler = Hydrus1DDATHandler(wpath)
        handler.read()
        
        i,wpath2 = tempfile.mkstemp()         
        handler2 = Hydrus1DDATHandler(wpath2, main=handler.main, profile=handler.profile)
        handler2.write()
        
        handler3 = Hydrus1DDATHandler(wpath2)
        handler3.read()
        self.assertEqual(handler.profile['ProfileDepth'], handler3.profile['ProfileDepth'])    

class TestSelectorHandler(unittest.TestCase):
    def test_read(self):
        wpath = './testdata/PRVE_swc_spring1_d6/SELECTOR.IN'
        handler = SELECTORHandler(wpath=wpath)
        handler.read()
        
        self.assertEqual(handler.LUnit, 'cm')
        self.assertEqual(handler.lWat, 't')
        self.assertEqual(handler.lSnow, 'f')



if __name__ == '__main__':
    unittest.main()