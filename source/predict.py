from trainer import *


if __name__ == '__main__':
    trainer = Trainer(args, module)
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")
    trainer.eval(test_dataset, mode="test")

