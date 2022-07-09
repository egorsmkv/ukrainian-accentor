import torch



# import
importer = torch.package.PackageImporter("accentor-lite.pt")
accentor = importer.load_pickle("uk-accentor", "model")

# run
test_words1 = ["словотворення", "архаїчний", "програма", "а-ля-фуршет"]

stressed_words = accentor.predict(test_words1, mode='stress')
plused_words = accentor.predict(test_words1, mode='plus')

print('With stress:', stressed_words)
print('With pluses:', plused_words)