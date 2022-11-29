from trainer import *
from main import args


if __name__ == '__main__':
    module = ISDFModule(args)
    tokenizer = load_tokenizer(args.pretrained_model)
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")
    trainer = Trainer(args, module)
    trainer.predict(test_dataset)

