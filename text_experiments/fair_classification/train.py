# coding=utf-8
import csv
import argparse
import logging
import os
import json
import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from tqdm import tqdm, trange
from transformers import AdamW
from model_utils import *
from models_transfert_style import *
from tokenizer_custom import ClassifTokenizer

try:
    from transformers import (get_linear_schedule_with_warmup, get_constant_schedule_with_warmup)
except:
    from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup
from models import *
from tensorboardX import SummaryWriter
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, args, dev, reny_ds=False):
        # TODO : this should be simplified before release + dict
        self.args = args
        logger.info("Loading dataset {}".format('Validation' if dev else 'Train'))
        suffix = 'dev' if dev else 'train'
        if args.classif:
            def open_classif(path):
                with open(path, 'r') as file:
                    lines = file.readlines()

                lines = [line.replace('\n', '') for line in lines]
                texts = []
                label_downstream = []
                label_protected = []
                for line in lines:
                    text = [int(w_id) for w_id in line.split('\t')[0].split(' ')]
                    text += (args.max_length - len(text)) * [args.padding_idx]
                    texts.append(text)  
                    label_downstream.append(int(line.split('\t')[1]))
                    label_protected.append(int(line.split('\t')[2]))
                return texts, label_downstream, label_protected

            file_name = 'x_val' if dev else 'x_train'
            if args.use_mention:
                file_path = os.path.join('processed_mention_splitted', file_name)
            elif args.use_bio:
                file_path = os.path.join('biais_bios', file_name)
            else:
                file_path = os.path.join('processed_sentiment_splitted', file_name)
            self.lines, self.label_downstream, self.label = open_classif(
                os.path.join('data/classification', file_path))


    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        if self.args.classif:
            return {'line': torch.tensor(self.lines[item], dtype=torch.long),
                    'label': torch.tensor(self.label[item], dtype=torch.long),
                    'downstream_labels': torch.tensor(self.label_downstream[item], dtype=torch.long)
                    }
        else:
            return {'line': torch.tensor(self.lines[item], dtype=torch.long),
                    'label': torch.tensor(self.label[item], dtype=torch.long)
                    }


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model):
    """ Train the model """
    tb_writer = SummaryWriter('runs/{}'.format(args.output_dir))
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size, drop_last=True)
    t_total = len(train_dataloader) * args.num_train_epochs
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)
    if args.no_scheduler:
        scheduler = get_constant_schedule_with_warmup(optimizer, args.warmup_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total * 100000)

    args.scheduler = scheduler
    args.optimizer = optimizer
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size = %f", args.batch_size)
    logger.info("  Total optimization steps = %f", t_total)
    if args.load_last_model:
        global_step = args.global_step
    else:
        global_step = 0
    epochs_trained = 0
    # tr_loss, logging_loss = 0.0, 0.0
    tr_loss_dic, logging_loss_dic = dict(), dict()
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproducibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            # model.training_all_except_encoder = False
            # if global_step % args.number_of_training_encoder == 0:
            # model.training_all_except_encoder = True
            if global_step > 40000 if args.classif else 100000:
                break
            inputs = batch['line'].to(args.device)
            labels = batch['label'].to(args.device)
            model.train()
            if args.classif:
                downstream_labels = batch['downstream_labels'].to(args.device)
                outputs = model(inputs, labels, downstream_labels)
            else:
                outputs = model(inputs, labels, teacher_ratio=1)
            dict_loss = outputs[0]
            if len(tr_loss_dic) == 0:
                logger.info('Initialization of training tensorboard dictionary')
                for key, value in dict_loss.items():
                    tr_loss_dic[key] = 0
                    logging_loss_dic[key] = 0

            for key, value in dict_loss.items():
                tr_loss_dic[key] += value.item()

            global_step += 1

            if global_step % args.eval_step == 0:
                results, sentences = evaluate(args, model)

                for key, value in results.items():
                    tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                tb_writer.add_text('Sentences', sentences, global_step)

                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                for key, value in tr_loss_dic.items():
                    tb_writer.add_scalar("train_{}".format(key),
                                         (tr_loss_dic[key]) / args.eval_step, global_step)

                logger.info("  lr = %5f", scheduler.get_lr()[0])
                for key, value in tr_loss_dic.items():
                    logger.info("  Training {} = %5f".format(key),
                                (tr_loss_dic[key]) / args.eval_step)

                for key, value in dict_loss.items():
                    tr_loss_dic[key] = 0
                    logging_loss_dic[key] = 0

                if args.reset_classif:
                    model.d_gamma.weights_init()
                    model.style_classifier.weights_init()

                # for key, value in tr_loss_dic.items():
                # logging_loss_dic[key] = tr_loss_dic[key]
            if global_step % args.save_step == 0:
                checkpoint_prefix = "checkpoint"
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                os.makedirs(output_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
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
    return 0


def evaluate(args, model, prefix=""):
    eval_dataset = TextDataset(args, True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, drop_last=True)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.batch_size)
    nb_eval_steps = 0
    model.training_all_except_encoder = False
    model.eval()
    eval_loss_dic = dict()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs = batch['line'].to(args.device)
        labels = batch['label'].to(args.device)

        with torch.no_grad():
            model = model.eval()
            if args.classif:
                downstream_labels = batch['downstream_labels'].to(args.device)
                outputs = model(inputs, labels, downstream_labels)
            else:
                outputs = model(inputs, labels, teacher_ratio=0)
            dict_loss = outputs[0]
            model = model.train()
            if len(eval_loss_dic) == 0:
                logger.info('Initialization of validation tensorboard dictionary')
                for key, value in dict_loss.items():
                    eval_loss_dic[key] = 0

            for key, value in dict_loss.items():
                eval_loss_dic[key] += dict_loss[key].item()
        nb_eval_steps += 1
        if nb_eval_steps == 20:
            break
    for key, value in dict_loss.items():
        eval_loss_dic[key] = eval_loss_dic[key] / (nb_eval_steps)
        logger.info("  Evaluation {} = %5f".format(key), value)
    index_sentences = [random.randint(0, inputs.size(0) - 1) for _ in range(10)]
    if not args.classif:
        try:
            try:
                outputs = outputs[index_sentences, :].tolist()

            except:
                outputs = outputs[-1][index_sentences, :].tolist()
            sentences_generated = [args.tokenizer.decode(output) for output in outputs]
            sentences_golden = [args.tokenizer.decode(inputs.tolist()[index]) for index in index_sentences]
            logger.info('Example of sentences')
            for i in range(10):
                logger.info('I : {}'.format(sentences_golden[i]))
                logger.info('G : {}'.format(sentences_generated[i]))
            sentences = ''
            for index, (s_gen, s_gol) in enumerate(zip(sentences_generated, sentences_golden)):
                sentences += 'Golden: {}\n\n\n Generated:{}\n\n\n ---------------------------------- \n\n\n'.format(
                    s_gol,
                    s_gen)
        except:
            outputs_g = outputs[-2][index_sentences, :].tolist()
            outputs_c = outputs[-1][index_sentences, :].tolist()
            sentences_generated = [args.tokenizer.decode(output) for output in outputs_g]
            sentences_corrupted = [args.tokenizer.decode(output) for output in outputs_c]
            sentences_golden = [args.tokenizer.decode(inputs.tolist()[index]) for index in index_sentences]
            logger.info('Example of sentences')
            for i in range(10):
                logger.info('I : {}'.format(sentences_golden[i]))
                logger.info('C : {}'.format(sentences_corrupted[i]))
                logger.info('G : {}'.format(sentences_generated[i]))
            sentences = ''
            for index, (s_gen, s_corr, s_gol) in enumerate(
                    zip(sentences_generated, sentences_corrupted, sentences_golden)):
                sentences += 'Golden: {}\n\n\n Corrupted: {}\n\n\n Generated:{}\n\n\n ---------------------------------- \n\n\n'.format(
                    s_gol, s_corr, s_gen)
        model.training_all_except_encoder = True
    else:
        sentences = ''
    return eval_loss_dic, sentences


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--dataset", default='yelp', type=str, help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default='debug',
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=5e-2, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=1000, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_length", default=43, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--eval_step", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_step", type=int, default=100, help="Save checkpoint every X updates steps.")
    parser.add_argument("--seed", type=int, default=23, help="random seed for initialization")  # change of seed

    # kernels
    parser.add_argument("--kernel_size", type=int, default=128, help="random seed for initialization")  # change of seed
    parser.add_argument("--loss_wasserstein", type=str, default="gaussian",
                        choices=["sinkhorn", "hausdorff", "energy", "gaussian", "laplacian"])
    parser.add_argument("--power", type=int, default=2, choices=[1, 2])
    parser.add_argument("--blur", type=float, default=0.05)

    parser.add_argument("--loss_type", type=str, default="wasserstein",
                        choices=['rao', 'frechet', 'js', 'wasserstein'])  # change of seed
    parser.add_argument("--mi_estimator_name", type=str, default="MINE",
                        choices=["NWJ", "MINE", "InfoNCE", "L1OutUB", "CLUB","KNIFE","DOE"])  # change of seed

    # Architecture
    parser.add_argument("--model", default='reny_desantanglement', help="random seed for initialization")
    parser.add_argument("--style_dim", type=int, default=32, help="random seed for initialization")
    parser.add_argument("--content_dim", type=int, default=480, help="random seed for initialization")
    parser.add_argument("--number_of_layers", type=int, default=2, help="random seed for initialization")
    parser.add_argument("--hidden_dim", type=int, default=512, help="random seed for initialization")
    parser.add_argument("--dec_hidden_dim", type=int, default=512, help="random seed for initialization")
    parser.add_argument("--dropout", type=float, default=0.5, help="random seed for initialization")
    parser.add_argument("--number_of_styles", type=int, default=2, help="random seed for initialization")
    parser.add_argument("--filter", type=int, default=12800, help="random seed for initialization")
    parser.add_argument("--mul_style", type=int, default=10, help="random seed for initialization")
    parser.add_argument("--adv_style", type=float, default=1, help="random seed for initialization")
    parser.add_argument("--mul_mi", type=float, default=1, help="random seed for initialization")
    parser.add_argument("--alpha", type=float, default=1.5, help="random seed for initialization")
    parser.add_argument("--ema_beta", type=float, default=0.99, help="random seed for initialization")
    parser.add_argument("--not_use_ema", action="store_true", help="random seed for initialization")
    parser.add_argument("--no_reny", action="store_true", help="random seed for initialization")
    parser.add_argument("--reny_training", type=int, default=5, help="random seed for initialization")
    parser.add_argument("--number_of_training_encoder", type=int, default=1, help="random seed for initialization")

    parser.add_argument("--add_noise", action="store_true", help="random seed for initialization")
    parser.add_argument("--noise_p", type=float, default=0.1, help="random seed for initialization")
    parser.add_argument("--number_of_perm", type=int, default=3, help="random seed for initialization")
    parser.add_argument("--load_seq2seq", action="store_true", help="random seed for initialization")
    parser.add_argument("--alternative_hs", action="store_true", help="random seed for initialization")
    parser.add_argument("--loading_path", type=str,
                        default='vanilla_seq2seq_h256/checkpoint-130000',
                        help="random seed for initialization")

    parser.add_argument("--no_minimization_of_mi_training", action="store_true")
    parser.add_argument("--special_clement", action="store_true")
    parser.add_argument("--complex_proj_content", action="store_true")
    parser.add_argument("--use_complex_classifier", action="store_true")
    parser.add_argument("--load_last_model", action="store_true")
    parser.add_argument("--no_scheduler", action="store_true")
    parser.add_argument("--reset_classif", action="store_true")

    parser.add_argument("--use_gender", action="store_true")
    parser.add_argument("--use_category", action="store_true")

    parser.add_argument("--classif", action="store_true")
    parser.add_argument("--use_mention", action="store_true")
    parser.add_argument("--use_bio", action="store_true")

    parser.add_argument("--use_complex_gamma_training", action="store_true")
    parser.add_argument("--use_club_estimation", action="store_true")
    parser.add_argument("--tight_training", action="store_true")

    parser.add_argument("--fine_tune_hkernel", action="store_true")

    parser.add_argument("--number_of_downstream_labels", type=int, default=2, help="random seed for initialization")
    args = parser.parse_args()

    if args.use_club_estimation:
        assert args.no_minimization_of_mi_training
    if args.use_bio:
        args.number_of_downstream_labels = 28
    if args.classif:
        args.max_length = 92
        if args.use_bio:
            args.max_length = 251
    elif args.use_gender:
        args.max_length = 150
    elif args.use_category:
        args.number_of_styles = 5
        args.max_length = 60
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Bounds hold only if considered reny > 1
    assert args.alpha > 1

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
    # Set seed
    set_seed(args)
    args.logger = logger
    logger.info("------------------------------------------ ")
    logger.info("model type = %s ", args.model)
    logger.info("------------------------------------------ ")

    if args.classif and not args.use_bio:
        tokenizer = ClassifTokenizer(
            'data/classification/processed_mention' if args.use_mention else 'data/classification/processed_sentiment')
    else:
        try:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        except:
            tokenizer = BertTokenizer.from_pretrained(
                '/gpfswork/rech/tts/unm25jp/transformers_models/bert-base-uncased/')

    args.sos_token = tokenizer.sep_token_id
    args.number_of_tokens = tokenizer.vocab_size
    args.tokenizer = tokenizer
    args.padding_idx = tokenizer.pad_token_id
    reny_dataset = TextDataset(args, False)
    reny_sampler = RandomSampler(reny_dataset)
    reny_dataloader = DataLoader(
        reny_dataset, sampler=reny_sampler, batch_size=args.batch_size, drop_last=True)

    if args.classif:
        ##################
        # Classification #
        ##################
        if args.model == 'mi_baseline':
            model = MIClassificationStyleEmdedding(args, reny_dataloader)
        elif args.model == 'knife':
            model = KernelClassificationStyleEmdedding(args, reny_dataloader)
        elif args.model == 'rao_regularizer':
            model = RaoClassificationStyleEmdedding(args, reny_dataloader)
        else:
            model = ClassificationStyleEmdedding(args, reny_dataloader)
    else:
        ##############
        # Generation #
        ##############
        if args.model == 'reny_desantanglement':
            model = RenySeq2Seq(args, reny_dataloader)
            if args.load_seq2seq:
                weights_encoder = model.encoder.embedding.weight.tolist()
                model_baseline = torch.load(os.path.join(args.loading_path, 'model.pt'), map_location=args.device)
                model.load_state_dict(model_baseline, strict=False)
                assert weights_encoder != model.encoder.embedding.weight.tolist()
        elif args.model == 'baseline_desantanglement':
            model = BaselineDisentanglement(args)
            if args.load_seq2seq:
                weights_encoder = model.encoder.embedding.weight.tolist()
                try:
                    logger.info("Loading %s", args.loading_path)
                    model_baseline = torch.load(os.path.join(args.loading_path, 'model.pt'), map_location=args.device)
                except:
                    logger.info("Error in Loading %s", args.loading_path)
                    args.loading_path = 'models/vanilla_seq2seq_h256/checkpoint-130000'
                    model_baseline = torch.load(os.path.join(args.loading_path, 'model.pt'), map_location=args.device)
                model.load_state_dict(model_baseline, strict=False)
                assert weights_encoder != model.encoder.embedding.weight.tolist()

        elif args.model == 'multi_dec':
            model = MultiDec(args, reny_dataloader)
        elif args.model == 'style_emb':
            if args.use_category:
                model = MultiStyleEmdedding(args, reny_dataloader)
            else:
                model = StyleEmdedding(args, reny_dataloader)
        elif args.model == 'dae':
            model = DAE(args, reny_dataloader)
        else:
            model = VanillaSeq2seq(args)
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    ##################################
    # Restart Training for Small GPU #
    ##################################
    if args.load_last_model:
        out_of_training = False
        try:
            all_directories = os.listdir(args.output_dir)
            checkpoints = []
            for directory in all_directories:
                checkpoints.append(int(directory.split('-')[-1]))
            checkpoints = sorted(checkpoints)
            args.global_step = int(checkpoints[-1])

            model.load_state_dict(
                torch.load(os.path.join(args.output_dir, 'checkpoint-{}'.format(args.global_step), 'model.pt'),
                           map_location=torch.device(args.device)))
            logger.info("Loading model %s", args.output_dir)
            logger.info("Epoch %s", 'checkpoint-{}'.format(args.global_step))

            if checkpoints[-1] > 120000:
                logger.info("Checkpoint is greater than 120000")
                out_of_training = True
                raise NotImplementedError

        except:
            args.global_step = 0
            if out_of_training:
                raise NotImplementedError
            logger.info("No model found")

    #########
    # Train #
    #########
    train_dataset = copy.copy(reny_dataset)
    train(args, train_dataset, model)
    logger.info(" Training Over ")


if __name__ == "__main__":
    main()
