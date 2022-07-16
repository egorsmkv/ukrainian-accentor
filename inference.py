import torch



# import
importer = torch.package.PackageImporter("accentor-lite.pt")
accentor = importer.load_pickle("uk-accentor", "model")

# run
text = "Я співаю веселу пісню в Україні"

stressed_words = accentor.process(text, mode='stress')
plused_words = accentor.process(text, mode='plus')

print('With stress:', stressed_words)
print('With pluses:', plused_words)