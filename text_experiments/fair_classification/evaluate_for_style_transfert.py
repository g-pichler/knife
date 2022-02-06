# coding=utf-8
import csv
import argparse
from argparse import ArgumentParser
import os
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from tqdm import tqdm, trange
import json

try:
    from transformers import (get_linear_schedule_with_warmup)
except:
    from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup
from models import *
from tensorboardX import SummaryWriter
from transformers import BertTokenizer
import copy
from transformers import *
from models_transfert_style import *
import logging
import torch
import random

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, args, dev, reny_ds=False):
        logger.info("Loading dataset {}".format('Validation' if dev else 'Train'))
        suffix = 'dev' if dev else 'train'

        if args.use_gender:
            if suffix == 'dev':
                suffix = "valid"
            with open(
                    'data/multiple_attribute/tensor_sentiment.{}_female'.format(
                        suffix),
                    'r') as file:
                csv_reader = csv.reader(file, delimiter=',')
                lines_pos = [[int(index) for index in line] for line in csv_reader if len(list(set(line))) > 5]
            with open('data/multiple_attribute/tensor_sentiment.{}_male'.format(
                    suffix),
                    'r') as file:
                csv_reader = csv.reader(file, delimiter=',')
                lines_neg = [[int(index) for index in line] for line in csv_reader if len(list(set(line))) > 5]

        else:
            with open('data/data_tensor_{}/tensor_sentiment.{}.1'.format(args.dataset, suffix),
                      'r') as file:
                csv_reader = csv.reader(file, delimiter=',')
                lines_pos = [[int(index) for index in line] for line in csv_reader]
            with open('data/data_tensor_{}/tensor_sentiment.{}.0'.format(args.dataset, suffix),
                      'r') as file:
                csv_reader = csv.reader(file, delimiter=',')
                lines_neg = [[int(index) for index in line] for line in csv_reader]

        labels = [1] * len(lines_pos) + [0] * len(lines_neg)
        lines = lines_pos + lines_neg
        temp = list(zip(labels, lines))
        random.seed(42)
        random.shuffle(temp)
        labels, lines = zip(*temp)

        if args.use_gender:
            filler_train = 40000
            self.lines = lines[:filler_train]
            self.label = labels[:filler_train]
        else:
            split = 30 * 128
            if dev and reny_ds:
                self.lines = lines[split:]
                self.label = labels[split:]
            elif dev:
                self.lines = lines[:split]
                self.label = labels[:split]
            else:
                self.lines = lines[:args.filter]
                self.label = labels[:args.filter]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return {'line': torch.tensor(self.lines[item], dtype=torch.long),
                'label': torch.tensor(self.label[item], dtype=torch.long)}


class ClassifierDataset(Dataset):
    def __init__(self, args, test_dataset, model):
        self.l_embeddings = []
        self.l_labels = []
        eval_sampler = SequentialSampler(test_dataset)
        eval_dataloader = DataLoader(
            test_dataset, sampler=eval_sampler, batch_size=args.batch_size, drop_last=True)

        # Eval!
        logger.info("***** Embedding evaluation *****")
        logger.info("  Batch size = %d", args.batch_size)
        model.eval()
        for batch in tqdm(eval_dataloader, desc="Converting The Dataset"):
            inputs = batch['line'].to(args.device)
            labels = batch['label'].to(args.device)
            with torch.no_grad():
                embeddings = model.predict_latent_space(inputs)
            self.l_embeddings.append(embeddings.cpu().detach())
            self.l_labels.append(labels.cpu())

        self.l_embeddings = torch.cat(self.l_embeddings, dim=1).permute(1, 0, 2).tolist()
        self.l_embeddings = [torch.tensor(i) for i in self.l_embeddings]
        self.l_labels = torch.cat(self.l_labels, dim=0).tolist()
        self.l_labels = [torch.tensor(i) for i in self.l_labels]

    def __len__(self):
        return len(self.l_labels)

    def __getitem__(self, item):
        return {'line': self.l_embeddings[item].float(),
                'label': self.l_labels[item].long()}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def clean_seq(args, outputs):
    pad_token = args.tokenizer.pad_token_id
    sep = args.tokenizer.sep_token_id
    cleaned_seq = []
    for seq in outputs:
        try:
            first_index_not_null = [i for i, x in enumerate(seq) if x == sep][1]
        except:
            first_index_not_null = 100
        seq_cleaned = []
        for i, x in enumerate(seq):
            if i < first_index_not_null:
                seq_cleaned.append(x)
            else:
                seq_cleaned.append(pad_token)
        cleaned_seq.append(seq_cleaned)
    return cleaned_seq


def train_classifer(args, classifier, train_dataset, dev_dataset, model):
    suffix = args.output_dir
    tb_writer = SummaryWriter('runs/{}'.format(suffix))
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size, drop_last=True)

    dev_sampler = RandomSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.batch_size, drop_last=True)

    t_total = len(train_dataloader) * args.num_train_epochs
    optimizer = AdamW(classifier.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)
    loss_fct = torch.nn.NLLLoss()

    args.scheduler = scheduler
    args.optimizer = optimizer
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size = %f", args.batch_size)
    logger.info("  Total optimization steps = %f", t_total)

    best_loss = 100000000
    global_step = 0
    epochs_trained = 0
    classifier.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproducibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            inputs = batch['line'].to(args.device)  # .permute(1, 0, 2)
            with torch.no_grad():
                inputs = model.predict_latent_space(inputs)
            labels = batch['label'].to(args.device)
            classifier.train()
            prediction = classifier(inputs)
            loss = loss_fct(prediction, labels.long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            classifier.zero_grad()

            tb_writer.add_scalar("train_loss", loss, global_step)
            tb_writer.add_scalar("train_lr", scheduler.get_lr()[0], global_step)

            global_step += 1
            if global_step % args.save_step == 0:
                dev_epoch_iterator = tqdm(dev_dataloader, desc="Dev Iteration")
                dev_loss = 0
                for dev_step, dev_batch in enumerate(dev_epoch_iterator):
                    inputs = dev_batch['line'].to(args.device)
                    with torch.no_grad():
                        inputs = model.predict_latent_space(inputs)
                    labels = dev_batch['label'].to(args.device)
                    classifier.eval()
                    prediction = classifier(inputs)
                    dev_loss += loss_fct(prediction, labels.long())
                if best_loss > dev_loss:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir)
                    os.makedirs(output_dir, exist_ok=True)
                    classifier_path = os.path.join(output_dir, 'classifier_latent_space.pt')
                    torch.save(classifier.state_dict(), classifier_path)
                    with open(os.path.join(output_dir, 'training_args.txt'), 'w') as f:
                        dict_to_save = copy.copy(args.__dict__)
                        for key, value in dict_to_save.items():
                            if value is None:
                                pass
                            elif isinstance(value, (bool, int, float)):
                                pass
                            elif isinstance(value, (tuple, list)):
                                pass
                            elif isinstance(value, dict):
                                pass
                            else:
                                dict_to_save[key] = 0
                        json.dump(dict_to_save, f, indent=2)
                    logger.info("Saving model checkpoint to %s", output_dir)
    # Load last best classifier :)
    logger.info("Last checkpoint %s", global_step)
    logger.info("Reloading Best Saved model at %s", classifier_path)
    classifier.load_state_dict(torch.load(classifier_path, map_location=torch.device(args.device)))


def evaluate_disantanglement(args, classifer, eval_dataset, model):
    loss_fct = torch.nn.NLLLoss()
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, drop_last=True)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.batch_size)
    classifer.eval()
    losses = []
    accuracies = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs = batch['line'].to(args.device)
        with torch.no_grad():
            inputs = model.predict_latent_space(inputs)
        labels = batch['label'].to(args.device)
        with torch.no_grad():
            prediction = classifer(inputs)
        loss = loss_fct(prediction, labels.long())
        losses.append(loss.item())
        accuracy = sum([i == j for i, j in zip(prediction.topk(1)[-1].squeeze(-1).tolist(), labels.tolist())]) / len(
            labels.tolist())
        accuracies.append(accuracy)
    loss = sum(losses) / len(losses)
    accuracy = sum(accuracies) / len(accuracies)
    logger.info("***** loss evaluation {} *****".format(loss))
    logger.info("***** accuracy evaluation {} *****".format(accuracy))
    os.makedirs(os.path.join('club_results_desantanglement', args.suffix), exist_ok=True)
    f = open(os.path.join('club_results_desantanglement', args.suffix, 'disantanglement.txt'), "w")
    f.write('Evaluation for disantanglement latent space: \n')
    f.write('accuracy\t:{}\n'.format(accuracy))
    f.write('loss\t:{}\n'.format(loss))


def test_style_transfert(args, eval_dataset, model, flip_label, train_dataset):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, drop_last=True)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %5d", len(eval_dataset))
    logger.info("  Batch size = %d", args.batch_size)
    nb_eval_steps = 0
    sentences_generated_ = []
    sentences_golden_ = []
    labels_ = []
    model.eval()
    j = 0
    style_pos, style_neg = 0, 0
    if args.model == 'dae':
        model = model.eval()
        for batch in tqdm(train_dataset, desc="Compute Style Vector"):
            inputs = batch['line'].to(args.device)
            labels = batch['label'].to(args.device)
            with torch.no_grad():
                style_neg_, style_pos_ = model.predict_style_space(inputs, labels)
                style_pos += torch.sum(style_pos_, dim=1)
                style_neg += torch.sum(style_neg_, dim=1)

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        j += 1
        inputs = batch['line'].to(args.device)
        labels = batch['label'].to(args.device)

        with torch.no_grad():
            model = model.eval()
            outputs = model.forward_transfert(inputs, 1 - labels if flip_label else labels, style_pos, style_neg)
            outputs = clean_seq(args, outputs.tolist())
            sentences_generated_ += outputs
            sentences_golden_ += inputs.tolist()
            labels_ += labels.tolist()
            nb_eval_steps += 1

    sentences_generated_ = [args.tokenizer.decode(output, skip_special_tokens=True) for output in sentences_generated_]
    sentences_golden_ = [args.tokenizer.decode(output, skip_special_tokens=True) for output in sentences_golden_]
    os.makedirs(os.path.join(args.sentences, args.suffix), exist_ok=True)
    with open(os.path.join(args.sentences, args.suffix, 'gen_{}.txt'.format(flip_label)), 'w') as file:
        file.writelines(['{}\n'.format(str(i)) for i in sentences_generated_])

    with open(os.path.join(args.sentences, args.suffix, 'label_gen_{}.txt'.format(flip_label)), 'w') as file:
        labels_w = [1 - i for i in labels_] if flip_label else labels_
        file.writelines(['{}\n'.format(str(i)) for i in labels_w])

    with open(os.path.join(args.sentences, args.suffix, 'golden_{}.txt'.format(flip_label)), 'w') as file:
        file.writelines(['{}\n'.format(str(i)) for i in sentences_golden_])

    with open(os.path.join(args.sentences, args.suffix, 'label_golden_{}.txt'.format(flip_label)), 'w') as file:
        labels_w = labels_
        file.writelines(['{}\n'.format(str(i)) for i in labels_w])


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--dataset", default='yelp', type=str, help="The input training data file (a text file).")
    parser.add_argument("--suffix", default='yelp', type=str, help="The input training data file (a text file).")
    parser.add_argument("--sentences", default='club_sentences_new/', type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--filter", default=100, type=int, help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default='debug',
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--batch_size", default=14, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_length", default=43, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--save_step", type=int, default=1000, help="random seed for initialization")
    parser.add_argument("--num_train_epochs", type=int, default=1000, help="random seed for initialization")
    parser.add_argument("--warmup_steps", type=int, default=0, help="random seed for initialization")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="random seed for initialization")
    parser.add_argument("--adam_epsilon", type=float, default=0.001, help="random seed for initialization")

    # Architecture Seq2Seq
    parser.add_argument("--model", default='style_emb', help="random seed for initialization")
    parser.add_argument("--model_path_to_load", default='checkpoint-10000',
                        help="random seed for initialization")
    parser.add_argument("--saving_result_file", default='test_transfert.txt', help="loading from path")

    # Classifier
    parser.add_argument("--path_classifier", default='classifier_vanilla_seq2seq_2_for_desantaglement_results',
                        help="loading from path")
    parser.add_argument("--checkpoints_path_classifier", default='checkpoint-173400', help="loading from path")

    # Architecture
    parser.add_argument("--style_dim", type=int, default=8, help="random seed for initialization")
    parser.add_argument("--content_dim", type=int, default=256, help="random seed for initialization")
    parser.add_argument("--number_of_layers", type=int, default=2, help="random seed for initialization")
    parser.add_argument("--hidden_dim", type=int, default=256, help="random seed for initialization")
    parser.add_argument("--dec_hidden_dim", type=int, default=136, help="random seed for initialization")
    parser.add_argument("--dropout", type=float, default=0.5, help="random seed for initialization")
    parser.add_argument("--number_of_styles", type=int, default=2, help="random seed for initialization")
    parser.add_argument("--mul_style", type=int, default=10, help="random seed for initialization")
    parser.add_argument("--adv_style", type=int, default=1, help="random seed for initialization")
    parser.add_argument("--mul_mi", type=float, default=1, help="random seed for initialization")
    parser.add_argument("--alpha", type=float, default=1.5, help="random seed for initialization")
    parser.add_argument("--ema_beta", type=float, default=0.99, help="random seed for initialization")
    parser.add_argument("--not_use_ema", action="store_true", help="random seed for initialization")
    parser.add_argument("--no_reny", action="store_true", help="random seed for initialization")
    parser.add_argument("--reny_training", type=int, default=2, help="random seed for initialization")
    parser.add_argument("--number_of_training_encoder", type=int, default=1, help="random seed for initialization")
    parser.add_argument("--add_noise", action="store_true", help="random seed for initialization")
    parser.add_argument("--noise_p", type=float, default=0.1, help="random seed for initialization")
    parser.add_argument("--number_of_perm", type=int, default=3, help="random seed for initialization")
    parser.add_argument("--alternative_hs", action="store_true", help="random seed for initialization")
    parser.add_argument("--no_minimization_of_mi_training", action="store_true")
    parser.add_argument("--special_clement", action="store_true")
    parser.add_argument("--use_gender", action="store_true")
    parser.add_argument("--complex_proj_content", action="store_true")

    # What to do
    parser.add_argument("--do_eval", action='store_true', help="loading from path")
    parser.add_argument("--do_train_classifer", action='store_true', help="loading from path")
    parser.add_argument("--do_test_reconstruction", action='store_true', help="loading from path")
    parser.add_argument("--do_test_transfer", action='store_true', help="loading from path")
    parser.add_argument("--use_complex_classifier", action='store_true', help="loading from path")

    # Metrics
    parser.add_argument("--model_metrics", default='model_for_metric_evaluation', help="loading from path")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    except:
        tokenizer = BertTokenizer.from_pretrained(
            '/gpfswork/rech/qsq/uwi62ct/transformers_models/bert-base-uncased/')
    args.sos_token = tokenizer.sep_token_id
    args.number_of_tokens = tokenizer.vocab_size
    args.tokenizer = tokenizer
    args.use_complex_classifier = True
    args.padding_idx = tokenizer.pad_token_id
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
    # Set seed
    set_seed(args)
    os.makedirs(os.path.join(args.sentences, args.suffix), exist_ok=True)
    os.makedirs(os.path.join('results_desantanglement', args.suffix), exist_ok=True)
    args_model = ArgumentParser()
    # args_model = parser_model.parse_args()
    with open(os.path.join(args.model_path_to_load, 'training_args.txt'), 'r') as f:
        args_model.__dict__ = json.load(f)

    args_model.sos_token = tokenizer.sep_token_id
    args_model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args_model.number_of_tokens = tokenizer.vocab_size
    args_model.tokenizer = tokenizer
    args_model.batch_size = args.batch_size
    args_model.padding_idx = tokenizer.pad_token_id

    logger.info("------------------------------------------ ")
    logger.info("model type = %s ", args.model)
    logger.info("------------------------------------------ ")

    if args.model == 'multi_dec':
        model = MultiDec(args_model, None)
    elif args.model == 'style_emb':
        try:
            if args_model.use_complex_classifier is None:
                args_model.use_complex_classifier = False
        except:
            args_model.use_complex_classifier = False
        model = StyleEmdedding(args_model, None)
    elif args.model == 'dae':
        model = DAE(args_model, None)
    weight_pr = model.encoder.embedding.weight.data.tolist()
    model.load_state_dict(torch.load(os.path.join(args.model_path_to_load, 'model.pt'),
                                     map_location=torch.device(args.device)))
    assert weight_pr != model.encoder.embedding.weight.data.tolist()
    args_model.content_dim = args_model.hidden_dim
    model.to(args.device)
    model.eval()

    logger.info("Model parameters %s", args)

    dev_dataset = TextDataset(args, True)

    if args.model == 'dae':
        train_dataset = TextDataset(args, False)
        eval_sampler = SequentialSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=eval_sampler, batch_size=args.batch_size, drop_last=True)
    else:
        train_dataloader = None

    if args.do_test_transfer:
        test_style_transfert(args, dev_dataset, model, True, train_dataloader)
        logger.info(" Test Over ")

    if args.do_test_reconstruction:
        test_style_transfert(args, dev_dataset, model, False, train_dataloader)
        logger.info(" Test Over ")

    if args.do_train_classifer:
        logger.info("------------------------------------------ ")
        logger.info("Training Classifier")
        logger.info("------------------------------------------ ")

        train_dataset = TextDataset(args, False)
        train_classifier_dataset = train_dataset  # ClassifierDataset(args, train_dataset, model)

        dev_classifier_dataset = dev_dataset  # ClassifierDataset(args, dev_dataset, model)

        classifier = Classifier(args_model.hidden_dim if args.model == 'multi_dec' else args_model.content_dim,
                                args_model.number_of_styles, args.use_complex_classifier).to(
            args.device)
        model.eval()
        train_classifer(args, classifier, train_classifier_dataset, dev_classifier_dataset, model)
        logger.info(" Training Over ")
        if args.do_eval:
            evaluate_disantanglement(args, classifier, dev_classifier_dataset, model)
            logger.info(" Testing evaluate_disantanglement over ")

    logger.info(" Program Over ")


if __name__ == "__main__":
    main()
