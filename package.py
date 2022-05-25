from accentor import Accentor, replace_accents
import torch



accentor = Accentor('./model/accentor.pt', './model/dict.txt')


with torch.package.PackageExporter("accentor-lite.pt") as exporter:
    # intern
    exporter.intern("accentor.**")

    # extern
    exporter.extern("numpy.**")
    # mock
    exporter.mock("pandas")

    #save
    exporter.save_pickle("uk-accentor", "model", accentor)
    exporter.save_pickle("uk-accentor", "replace_accents", replace_accents)

print("ok")