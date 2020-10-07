# encoding=utf-8
import jieba

# 基于jieba的分词
seq_list = jieba.cut("周腾专注人工智能技术", cut_all=False)
print("Default Mode: " + "/ ".join(seq_list))

jieba.add_word("人工智能技术")
seq_list = jieba.cut("周腾专注人工智能技术", cut_all=False)
print("Default Mode: " + "/ ".join(seq_list))
