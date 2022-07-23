from accentor import Accentor
import torch



accentor = Accentor('./model/accentor.pt', './model/dict.txt')


with torch.package.PackageExporter("ukrainian_accentor/accentor-lite.pt") as exporter:
    # intern
    exporter.intern("accentor.**")
    exporter.intern("word_tokenizer.**")

    # extern
    exporter.extern("numpy.**")
    exporter.extern("six.**")
    # mock
    exporter.mock("pandas")

    #save
    exporter.save_pickle("uk-accentor", "model", accentor)

print("ok")