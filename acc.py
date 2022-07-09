import torch
from tqdm import tqdm


# import
importer = torch.package.PackageImporter("accentor-lite.pt")
accentor = importer.load_pickle("uk-accentor", "model")

# run
words1 = []
words2 = []
with open("./model/dict.txt") as fil:
    lines = fil.readlines()
    for line in lines:
        word2 = line.strip()
        word1 = word2.replace(chr(769),"")
        words1.append(word1)
        words2.append(word2)


period = 10000
total = 0
right = 0
for i in tqdm(range(0, len(words1),period)):
    word1 = words1[i:i+period]
    word2 = words2[i:i+period]

    total += period
    predicted = accentor.predict(word1, mode='stress')
    for j in range(len(word1)):
        if (predicted[j] == word2[j]):
            right += 1


print('Acc:', right / total)