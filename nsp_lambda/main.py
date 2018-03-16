import data
import mdl_building
import mdl_running
import os
import jpype
import config
import ast

if __name__ == '__main__':
    jar_path = os.path.join(config.opts.lib, 'lDCS_convertor.jar')
    jpype.startJVM(jpype.getDefaultJVMPath(), '-ea',
                   '-Djava.class.path=%s' % jar_path)
    data.load()
    if config.opts.train:
    	mdl_building.train()
    mdl_running.test('_whole')

    jpype.shutdownJVM()

