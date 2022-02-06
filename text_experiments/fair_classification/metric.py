import torch
import math
from tqdm import tqdm
import fasttext
import sacrebleu
import os
import argparse


class StyleTransfertMetric:
    def __init__(self, args, use_w_overlap, use_style_accuracy, use_ppl, style_classifier=None, lm=None,
                 tokenizer=None, not_use_vector=False):
        self.use_w_overlap = use_w_overlap
        self.use_style_accuracy = use_style_accuracy
        self.use_ppl = use_ppl
        self.style_classifier = style_classifier
        self.lm = lm
        self.args = args
        self.tokenizer = tokenizer
        if self.use_style_accuracy and style_classifier is None:
            print('You need a style classifier')
            raise NotImplementedError

        if self.use_ppl and lm is None:
            print('You need a lm')
            raise NotImplementedError
        print('Vectors Loaded')
        self.filer = self.extract_filters()
        print('Loading Vectors')
        self.vectors = self.load_vectors('wiki-news-300d-1M.vec')  # ("wiki-news-300d-1M.vec")

    """
    All input of metrics is [sent1,sent2,....,sentN] [gold1,gold2,....,goldN] where senti = [w_1,...w_k]
    """

    def extract_filters(self):
        try:
            with open('bow_features/negative-words.txt', 'r', encoding='utf-8') as file:
                lines = file.readlines()
            with open('bow_features/positive-words.txt', 'r', encoding='utf-8') as file:
                lines += file.readlines()
            with open('bow_features/stopwords.txt', 'r', encoding='utf-8') as file:
                lines += file.readlines()
        except:
            with open('../../bow_features/negative-words.txt', 'r', encoding='utf-8') as file:
                lines = file.readlines()
            with open('../../bow_features/positive-words.txt', 'r', encoding='utf-8') as file:
                lines += file.readlines()
            with open('../../bow_features/stopwords.txt', 'r', encoding='utf-8') as file:
                lines += file.readlines()
        filters_words = [line.replace('\n', '') for line in lines]
        return filters_words

    def load_vectors(self, fname):
        try:
            with open(fname, 'r', encoding='utf-8') as file:
                lines = file.readlines()
        except:
            with open('./wiki-news-300d-1M.vec', 'r', encoding='utf-8') as file:
                lines = file.readlines()

        lines_ = lines[1:]
        lines = [line.split(' ')[1:] for line in lines_]
        keys = [line.split(' ')[0] for line in lines_]
        count = 0
        dict_ = dict()
        for key in tqdm(keys):
            dict_[key.lower()] = [float(i) for i in lines[count]]
            count += 1

        return dict_

    def compute_cosinus_similarity(self, generated_sentences, golden_sentences):
        score = []
        for gen_s, gold_s in zip(generated_sentences, golden_sentences):
            vect_gen = []
            vect_gold = []
            for wgen in gen_s:
                try:
                    vect_gen.append(self.vectors[wgen.lower()])
                except:
                    pass
            vect_gen = [sum(x) for x in zip(*vect_gen)]

            for wgen in gold_s:
                try:
                    vect_gold.append(self.vectors[wgen.lower()])
                except:
                    pass
            vect_gold = [sum(x) for x in zip(*vect_gold)]

            num = sum([a * b for a, b in zip(vect_gen, vect_gold)])
            denom = (sum([a ** 2 for a in vect_gen]) * sum([b ** 2 for b in vect_gold])) ** (0.5)
            if num == 0 or denom == 0:
                score.append(0)
            else:
                score.append(num / denom)
        if len(score) == 0:
            return 0
        return sum(score) / len(score)

    def compute_blue_score(self, generated_sentences, golden_sentences):
        generated_sentences = self.remove_pad_and_sep(generated_sentences)
        generated_sentences = [' '.join(i) for i in generated_sentences]
        golden_sentences = self.remove_pad_and_sep(golden_sentences)
        golden_sentences = [[' '.join(i) for i in golden_sentences]]
        bleu_score = sacrebleu.corpus_bleu(generated_sentences, golden_sentences).score  # corpus_bleu
        return bleu_score

    def clean(self, word_list):
        cleaned_ = []
        for word in word_list:
            if word not in self.filer:
                cleaned_.append(word)
        if cleaned_ == []:
            cleaned_ = ['']
        return cleaned_

    def compute_w_overlap(self, generated_sentences, golden_sentences):
        # split with space
        assert self.use_w_overlap
        generated_sentences = self.remove_pad_and_sep(generated_sentences)
        golden_sentences = self.remove_pad_and_sep(golden_sentences)

        w_overlap = []
        for gen, gold in zip(generated_sentences, golden_sentences):
            word_gen = list(set(gen))
            word_gold = list(set(gold))
            # remove the words
            word_gen = self.clean(word_gen)
            word_gold = self.clean(word_gold)

            inter = len([value for value in word_gold if value in word_gen])
            union = len(list(set(word_gen + word_gold)))
            w_overlap.append(inter / union)
        return sum(w_overlap) / len(w_overlap)

    def compute_style_accuracy(self, generated_sentences, golden_sentences, labels):
        assert self.use_style_accuracy
        generated_sentences = self.remove_pad_and_sep(generated_sentences)
        golden_sentences = self.remove_pad_and_sep(golden_sentences)
        gold_labels = []
        gen_labels = []
        for gen, gold in zip(generated_sentences, golden_sentences):
            gold = ' '.join(gold)
            gen = ' '.join(gen)
            gold_labels.append(int(self.style_classifier.predict(gold)[0][0].replace('__label__', '')))
            gen_labels.append(int(self.style_classifier.predict(gen)[0][0].replace('__label__', '')))
        acc_gen = sum(1 for x, y in zip(gen_labels, labels) if x == y) / float(len(gen_labels))
        acc_gold = sum(1 for x, y in zip(gold_labels, labels) if x == y) / float(len(gold_labels))
        return acc_gen, acc_gold

    def compute_ppl(self, generated_sentences, golden_sentences):
        def score(sentences, model):
            perplexities = []
            tensor_inputs = []
            for sentence in tqdm(sentences, desc='lm'):
                tokenize_input = self.tokenizer.tokenize(sentence)
                tokenized_string = self.tokenizer.encode(tokenize_input, max_length=self.args.max_length)
                tokenized_string += [self.tokenizer.unk_token_id] * (self.args.max_length - len(tokenized_string))
                tensor_input = torch.tensor([tokenized_string]).to(self.args.device)
                tensor_inputs.append(tensor_input)

            tensor_inputs = torch.cat(tensor_inputs, dim=0)

            for k in range(len(sentences) // self.args.batch_size):
                tensors = tensor_inputs[k * self.args.batch_size:(k + 1) * self.args.batch_size, :].to(
                    self.args.device)
                loss = model(tensors, labels=tensors)[0].item()
                perplexities.append(math.exp(loss / len(tokenize_input)))
            return sum(perplexities) / len(perplexities)

        gen = self.remove_pad_and_sep(generated_sentences)
        gold = self.remove_pad_and_sep(golden_sentences)

        ppl_generated_sentences = score([' '.join(i) for i in gen], self.lm)
        ppl_golden_sentences = score([' '.join(i) for i in gold], self.lm)
        return ppl_generated_sentences, ppl_golden_sentences

    def remove_pad_and_sep(self, sentences):
        cleaned_sentences = []
        filter_words = ['[SEP]', '[PAD]', '[CLS]']
        for sentence in sentences:
            words = [word for word in sentence if word not in filter_words]
            cleaned_sentences.append(words)
        return cleaned_sentences


if __name__ == '__main__':
    def clean_sentence(lines):
        filter_words = ['[SEP]', '[PAD]']
        cleaned_lines = []
        for line in lines:
            line_splitted = line.replace('\n', '').split(' ')
            line_splitted_ = []
            for word in line_splitted:
                if word not in filter_words:
                    line_splitted_.append(word)
            cleaned_lines.append(line_splitted_)
        return cleaned_lines


    label_file = 'label_gen_True.txt'
    text_file = 'gen_True.txt'
    with open(label_file, 'r') as file:
        labels = file.readlines()

    labels = [int(i.replace('\n', '')) for i in labels]

    with open(text_file, 'r') as file:
        lines = clean_sentence(file.readlines())

    dataset = 'yelp'
    model_metrics = 'model_for_metric_evaluation_sentiment'
    style_classifier = fasttext.load_model(
        '{}.bin'.format(os.path.join(model_metrics, '{}_fastText'.format(dataset), 'fastText')))
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=150, help="For distant debugging.")
    parser.add_argument("--batch_size", type=int, default=2, help="For distant debugging.")
    args = parser.parse_args()
    args.device = torch.device('cpu')

    metric = StyleTransfertMetric(args, use_w_overlap=False, use_style_accuracy=True, use_ppl=False,
                                  style_classifier=style_classifier, lm=None, tokenizer=None)
    # style_accuracy = metric.compute_style_accuracy(lines, lines, labels)
    # '''' dummy test '''
    # import fasttext
    #
    # model = fasttext.load_model('{}.bin'.format(os.path.join('amazon_fastText', 'fastText')))
    # metric = Metric(True, True, False, False, model)
    generated_sentences = ['Hello how do you do', '[SEP] Hi ! [PAD]', 'Hello how do you do', '[SEP] Hi ! [PAD]',
                           'Hello how do you do', '[SEP] Hi ! [PAD]', 'Hello how do you do', '[SEP] Hi ! [PAD]']
    gold_sentences = ['Hello how do you do', 'Hi ']
    # labels = [0, 1]
    # # print(metric.compute_style_accuracy(generated_sentences, gold_sentences, labels))
    # # print(metric.compute_w_overlap(generated_sentences,gold_sentences))
    # # print(metric.remove_pad_and_sep(generated_sentences))
    #
    import math
    import torch
    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    #
    # # Load pre-trained model (weights)
    model = GPT2LMHeadModel.from_pretrained('model_for_metric_evaluation_sentiment/gpt_2_for_amazon')
    model.eval()
    # # Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('model_for_metric_evaluation_sentiment/gpt_2_for_amazon')

    metric = StyleTransfertMetric(args, use_w_overlap=False, use_style_accuracy=True, use_ppl=True,
                                  style_classifier=style_classifier, lm=model, tokenizer=tokenizer)
    metric.compute_ppl(generated_sentences, generated_sentences)
    #
    # print(score(['Hello how do you do', '[SEP] Hi ! [PAD]']))
    # print(score(['Hello how do you do', 'Hi how are you ?']))
