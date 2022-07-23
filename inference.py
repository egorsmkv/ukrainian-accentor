import ukrainian_accentor as accentor


text = "Я співаю веселу пісню в Україні"

stressed_words = accentor.process(text, mode='stress')
plused_words = accentor.process(text, mode='plus')

print('With stress:', stressed_words)
print('With pluses:', plused_words)