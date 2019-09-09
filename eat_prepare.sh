eat_format=$1
if [[ $eat_format = new ]]; then
    suf=eatx
    src_seq_length=120
elif [[ $eat_format = combined ]]; then
    suf=eatc
    src_seq_length=180
elif [[ $eat_format = old ]]; then
    suf=eat
    src_seq_length=60
else
    echo unsupported eat format
    exit 1
fi

# Prepare eat_en -> plain_en, for both train and dev.
python preprocess.py -train_src data/train.en.${suf} -train_tgt data/train.en.plain -valid_src data/dev.en.${suf} -valid_tgt data/dev.en.plain -save_data data/${suf}_en-plain_en -src_vocab_size 7500 -tgt_vocab_size 7500 --src_seq_length ${src_seq_length}
# Prepare eat_de for train. Note that plain_de is selected for target even though it might not be used for now.
python preprocess.py -train_src data/train.de.${suf} -train_tgt data/train.de.plain -valid_src data/dev.de.${suf} -valid_tgt data/dev.de.plain -save_data data/${suf}_de-plain_de -src_vocab_size 15000 -tgt_vocab_size 15000 --src_seq_length ${src_seq_length}
# Prepare eat_de -> plain_en. This is the crosslingual dataset used for validation. Note that source and target vocabs are reused.
python preprocess.py -train_src data/dev.de-en.de.eat -train_tgt data/dev.de-en.en.plain -save_data data/dev.de-en.eat-plain --src_vocab data/${suf}_de-plain_de.vocab.pt --tgt_vocab data/${suf}_en-plain_en.vocab.pt --force_tgt_vocab --src_seq_length 60

