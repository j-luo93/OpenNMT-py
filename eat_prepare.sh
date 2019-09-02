# Prepare eat_en -> plain_en, for both train and dev.
python preprocess.py -train_src data/train.en.eat -train_tgt data/train.en.plain -valid_src data/dev.en.eat -valid_tgt data/dev.en.plain -save_data data/eat_en-plain_en -src_vocab_size 7500 -tgt_vocab_size 7500
# Prepare eat_de for train. Note that plain_de is selected for target even though it might not be used for now.
python preprocess.py -train_src data/train.de.eat -train_tgt data/train.de.plain -save_data data/eat_de-plain_de -src_vocab_size 15000 -tgt_vocab_size 15000
# Prepare eat_de -> plain_en. This is the crosslingual dataset used for validation. Note that source and target vocabs are reused.
python preprocess.py -train_src data/dev.de-en.de.eat -train_tgt data/dev.de-en.en.plain -save_data data/dev.de-en.eat-plain --src_vocab data/eat_de-plain_de.vocab.pt --tgt_vocab data/eat_en-plain_en.vocab.pt --force_tgt_vocab
