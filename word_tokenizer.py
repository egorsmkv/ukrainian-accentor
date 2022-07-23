import re
import six



ACCENT = six.unichr(769)
WORD_TOKENIZATION_RULES = re.compile(r"""
[\w""" + ACCENT + """]+://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+
|[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+.[a-zA-Z0-9-.]+
|[0-9]+-[а-яА-ЯіїІЇ'’`""" + ACCENT + """]+
|[+-]?[0-9](?:[0-9,.-]*[0-9])?
|[\w""" + ACCENT + """](?:[\w'’`-""" + ACCENT + """]?[\w""" + ACCENT + """]+)*
|[\w""" + ACCENT + """].(?:\[\w""" + ACCENT + """].)+[\w""" + ACCENT + """]?
|["#$%&*+,/:;<=>@^`~…\\(\\)⟨⟩{}\[\|\]‒–—―«»“”‘’'№]
|[.!?]+
|-+
""", re.X | re.U)


WORD_MATCH_RULES = re.compile("(?=[а-яА-ЯіїІЇ'’`" + ACCENT + "])(?=.*[аеєиіїоуюяАЕЄИІЇОУЮЯ].*)")



def tokenize(text: str) -> list:
    finditer = re.finditer(WORD_TOKENIZATION_RULES, text)
    words = []
    for word in finditer:
        if WORD_MATCH_RULES.match(word.group()):
            words.append([word.group(), word.span()])
    return words


def detokenize(text: str, words: list) -> str:
    de_text = ""
    start_idx = 0
    for word, span in words:
        de_text += text[start_idx:span[0]]
        de_text += word
        start_idx = span[1]
    de_text += text[start_idx:-1]
    return de_text