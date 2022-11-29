from trainer import *


if __name__ == '__main__':
    module = ISDFModule(args)
    trainer = Trainer(args, module)
    trainer.predict()

