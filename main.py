import sys
import importlib
import options

if __name__=="__main__":
    opt = options.set(arg=sys.argv[1:])

    model = importlib.import_module("model.{}".format(opt.model))
    m = model.Model(opt)
    m.build_network(opt)
    m.set_optimizer(opt)
    m.create_dataset(opt)
    m.train(opt)

    m.end_process(opt)