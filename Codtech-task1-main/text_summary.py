import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

nltk.download('punkt')

# Long article text
text = """
Climate change refers to significant, long-term changes in the global climate. It is primarily driven by human activities, especially the burning of fossil fuels such as coal, oil, and natural gas. These activities release large amounts of greenhouse gases like carbon dioxide (CO2) and methane into the atmosphere, which trap heat and cause the Earth's temperature to rise. This phenomenon is known as global warming, and it leads to a variety of environmental impacts.

One major consequence of climate change is the increase in extreme weather events. Hurricanes, floods, droughts, and wildfires are becoming more frequent and severe due to changes in temperature and precipitation patterns. These events can destroy homes, disrupt food and water supplies, and threaten lives.

Melting glaciers and ice caps are another alarming effect. As these ice bodies shrink, sea levels rise, threatening coastal communities around the world. Additionally, the warming of oceans harms marine life, including coral reefs which are dying at an unprecedented rate due to ocean acidification and heat stress.

Addressing climate change requires global cooperation and strong policies. Countries must transition to renewable energy sources like solar, wind, and hydro power. Governments, businesses, and individuals all have a role to play in reducing carbon footprints, conserving energy, and protecting natural ecosystems.

If the world fails to act decisively, future generations may face unlivable conditions. But with urgent and sustained efforts, we can slow the rate of climate change and build a more sustainable future for all.
"""

# Summarization using Sumy (Extractive)
parser = PlaintextParser.from_string(text, Tokenizer("english"))
summarizer = LsaSummarizer()
summary = summarizer(parser.document, sentences_count=3)

print(text) #To print the text paragraph/article as it is.
print("\n")
print("📘 Extractive Summary:\n")
for sentence in summary:
    print("•", sentence)
