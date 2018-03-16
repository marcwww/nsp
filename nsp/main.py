import data
import mdl_running
import mdl_building
import config

if __name__ == '__main__':

    data.load()
    if config.opts.train:
    	mdl_building.train()
    mdl_running.test('_whole')
