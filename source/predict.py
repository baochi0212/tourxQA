from trainer import *
import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from utils import MODEL_DICT, get_intent_labels, get_slot_labels, init_logger, load_tokenizer
from trainer import Trainer_IDSF, Trainer_QA

logger = logging.getLogger(__name__)


def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, "training_args.bin"))


def load_model(args, device):
    # Check whether model exists
    path = args.idsf_model_dir if args.module_role == "IDSF" else args.qa_model_dir
    task = 'idsf' if args.module_role == "IDSF" else 'qa'
    if not os.path.exists(path):
        raise Exception("Model doesn't exists! Train first!")

    try:
        load_dir = path + f"/{args.model_type}_{int(args.n_epochs)}_{args.learning_rate}.pt"
        config, _, model = MODEL_DICT[args.model_type] if task == 'idsf' else QA_DICT[args.model_type]
        config = config.from_pretrained(args.pretrained_model)
        if task == 'idsf':
            model = model(config, args=args, intent_label_lst=get_intent_labels(args), slot_label_lst=get_slot_labels(args))
        else:
            model = model(config, args=args)
        model.load_state_dict(

            torch.load(load_dir)
        )
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except Exception:
        raise Exception("Some model files might be missing...")

    return model


def read_input_file(args):
    lines = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            words = line.split()
            if args.module_role == 'IDSF':
                lines.append(words)
            else:
                lines.append(line)

    return lines


def convert_input_file_to_tensor_dataset(
    lines,
    args,
    tokenizer,
    pad_token_label_id,
    cls_token_segment_id=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []

    for words in lines:
        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_label_mask.extend([pad_token_label_id + 1] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[: (args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[: (args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)

    return dataset


def predict_IDSF(args):
    # load model and args
    device = args.device
    model = load_model(args, device)
    logger.info(args)

    intent_label_lst = get_intent_labels(args)
    slot_label_lst = get_slot_labels(args)

    # Convert input file to TensorDataset
    pad_token_label_id = args.ignore_index
    tokenizer = load_tokenizer(args)
    lines = read_input_file(args) if not args.text_question else [args.text_question]
    dataset = convert_input_file_to_tensor_dataset(lines, args, tokenizer, pad_token_label_id)

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size)

    all_slot_label_mask = None
    intent_preds = None
    slot_preds = None

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "intent_label_ids": None,
                "slot_labels_ids": None,
            }
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            _, (intent_logits, slot_logits) = outputs[:2]

            # Intent Prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)

            # Slot prediction
            if slot_preds is None:
                if args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()
                all_slot_label_mask = batch[3].detach().cpu().numpy()
            else:
                if args.use_crf:
                    slot_preds = np.append(slot_preds, np.array(model.crf.decode(slot_logits)), axis=0)
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)
    #confusion
    prob = torch.max(nn.functional.softmax(intent_logits, -1), axis=1).item()
    intent_preds = np.argmax(intent_preds, axis=1)

    if not args.use_crf:
        slot_preds = np.argmax(slot_preds, axis=2)

    slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
    slot_preds_list = [[] for _ in range(slot_preds.shape[0])]

    for i in range(slot_preds.shape[0]):
        for j in range(slot_preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

    # Write to output file
    with open(args.output_file, "w", encoding="utf-8") as f:
        for words, slot_preds, intent_pred in zip(lines, slot_preds_list, intent_preds):
            line = ""
            for word, pred in zip(words, slot_preds):
                if pred == "O":
                    line = line + word + " "
                else:
                    line = line + "[{}:{}] ".format(word, pred)
            if not args.text_question:
                f.write("<{}> -> {}\n".format(intent_label_lst[intent_pred], line.strip()))
            else: 
                #return the label and slot filling for question
                with open(args.text_question_log_dir, 'w') as f:
                    f.write("prob <{}> -> <{}> -> {}\n".format(prob, intent_label_lst[intent_pred], line.strip()))

    logger.info("Prediction Done!")

def predict_QA(args):
    with torch.no_grad():
        #model
        model = load_model(args, args.device)


        tokenizer = load_tokenizer(args)
        #lines format: <Q @@ C>  
        lines = read_input_file(args) if not args.text_question else [args.text_question]
        
        for line in lines:
            q, c = line.split("@@")
            input = tokenizer(q.strip(), c.strip(), return_tensors='pt',
                max_length=args.qa_max_length,
                truncation="only_second",
                return_offsets_mapping=True,
                padding="max_length",
                return_token_type_ids=True,)

        
    
            inputs = {
                "input_ids": input.input_ids,
                "attention_mask": input.attention_mask,
                "token_type_ids": input.token_type_ids,
            }
            inputs = dict([(key, value.to(args.device)) for key, value in inputs.items()])
            #inputs for calculating validation loss
            outputs = model(**inputs)
            start, end = outputs
            # print("PREDICTION", start, end)
            # print("????", inputs["input_ids"][0])
            # print("FULL", tokenizer.decode(inputs["input_ids"]))
            pred = ' '.join(tokenizer.decode(inputs["input_ids"][0]).split()[start:end+1])
            if not args.text_question:
                with open(args.output_file, 'a') as f:
                    f.write(pred + '\n')
            else:
                with open(args.text_question_log_dir, 'w') as f:
                    f.write("<{}> -> {}\n".format(intent_label_lst[intent_pred], line.strip()))
        logger.info("Prediction done")


if __name__ == "__main__":
    '''
    we can both test and test dataset/ sample_input file .txt/ directly input to console !!!
    '''
    if args.text_question:
        predict_IDSF(args)
        # predict_QA(args)
    elif args.module_role == 'IDSF':
        if args.predict_task == "test example":
            init_logger()
            predict_IDSF(args)
        if args.predict_task == "test dataset":
            module = IDSFModule(args)
            tokenizer = load_tokenizer(args)
            test_dataset = load_and_cache_examples(args, tokenizer, mode="test")
            trainer = Trainer_IDSF(args, module)
            trainer.predict(test_dataset)
  
    else:
        init_logger()
        predict_QA(args)
    
        
        


