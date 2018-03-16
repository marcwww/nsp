from optparse import OptionParser
import os

home=os.path.abspath('..')

data_dir=os.path.join(home,'data')
free917_dir=os.path.join(data_dir,'free917')
res_dir=os.path.join(home,'res')
res_lambda=os.path.join(res_dir,'lambda')
mdl_dir=os.path.join(home,'mdl')
mdl_lambda=os.path.join(mdl_dir,'lambda')
lib_dir=os.path.join(home,'lib')
embs=os.path.join(data_dir,'embs')
emb=os.path.join(embs,'glove.6B.100d.txt')

opt_parser=OptionParser()
opt_parser.add_option('--data',dest='data',metavar='FILE',default=free917_dir)
opt_parser.add_option('--emb',dest='emb',metavar='FILE',default=emb)
opt_parser.add_option('--result',dest='result',metavar='FILE',default=res_lambda)
opt_parser.add_option('--model',dest='model',metavar='FILE',default=mdl_lambda)
opt_parser.add_option('--lib',dest='lib',metavar='FILE',default=lib_dir)
opt_parser.add_option('--dim_word',dest='dim_word',type='int',default=50)
opt_parser.add_option('--dim_term',dest='dim_term',type='int',default=50)
opt_parser.add_option('--dim_h',dest='dim_h',type='int',default=150)
opt_parser.add_option('--dim_act',dest='dim_act',type='int',default=50)
opt_parser.add_option('--nn_layers',dest='nn_layers',type='int',default=1)
opt_parser.add_option('--dropout',dest='dropout',type='float',default=0.5)
opt_parser.add_option('--optimizer',dest='optimizer',type='string',default='momentum')
opt_parser.add_option('--train',dest='train',default=False)
opt_parser.add_option('--epochs',dest='epochs',type='int',default=200)
opt_parser.add_option('--print_every',dest="print_every",type="int",default=500)


(opts,_)=opt_parser.parse_args()