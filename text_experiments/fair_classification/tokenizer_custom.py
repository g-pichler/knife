import os


class ClassifTokenizer:
    def __init__(self, folder_vocab_path):
        self.vocab_path = os.path.join(folder_vocab_path, 'vocab')
        with open(self.vocab_path, 'r', encoding="ISO-8859-1") as file:
            lines = file.readlines()

        lines = [line.replace('\n', '') for line in lines]
        self.id2text = self.convert_id_to_text(lines)
        self.text2id = self.convert_text_to_id(lines)

        self.pad_token_id = len(lines) + 1
        self.sep_token_id = len(lines) + 2
        self.vocab_size = len(lines) + 2
        self.cls_token_id = len(lines) + 3

    def __len__(self):
        return self.cls_token_id

    def convert_id_to_text(self, lines):
        return {k: v for v, k in enumerate(lines)}

    def convert_text_to_id(self, lines):
        return {v: k for v, k in enumerate(lines)}

    def decode(self, l_sentences):
        text_sentences = []
        for sentence in l_sentences:
            cur_sentence = []
            for word in sentence:
                cur_sentence.append(self.text2id[word])
            text_sentences.append(' '.join(cur_sentence))
        return text_sentences


if __name__ == '__main__':
    folder_vocab = 'data/classification/processed_mention'
    tokenizer = ClassifTokenizer(folder_vocab)
    print(tokenizer.decode([[134, 2445, 354, 66], [12, 354, 666]]))
