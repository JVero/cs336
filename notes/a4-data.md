# Assignment 4: Data

Handout: `assignment4-data/cs336_assignment4_data.pdf`

One heading per graded problem. Written answers, experiment numbers, plots, and things I want to remember go here.

## look_at_cc
a. The very first URL in it is "http://020bld.cn.vauofnj.cn/html/991f998999.html", which is not currently available. https://vauofnj.cn/ is, however. From the HTML I cannot tell what it is because it is in Mandarin Chinese.
b. The text that was filtered out by the extractor is html-specific text. Some valid sentences that the extractor might have filtered out are comments from the developers (html has comments), the javascript, as well as text that the extractor might have confused with HTML. ( < 3 > looks like it could either be an html tag or an ascii emoticon. ). Scrolling to the first english example, you notice that most of it isn't prose but is menus. Interestingly a lot of the information is text from buttons so those are more about actions ("Skip to main content"), rather than prose. One thing thats filtered out by the extractor is the heirarchy of the information. Most written text is written top to bottom but websites have nesting, z-stacking, and other things that require more freedom of orientation. What might go "wrong" is that the mandarin might make Claude feel like it needs to split between english and Chinese. Conversely keeping data in its native language can help llms implicitly learn cultural norms, since the data is kept in its native language. Also, after translating it myself, it is littered with SEO phrasing that isn't correct or good writing.
c. The first page describes how to get lily stamen stains out of clothes and floor. This is a useful thing for it to know, to help people who have that issue commonly. Specifically in a vacuum this would help Chinese people the most because it doesn't necessarily have the bridge to explain this information in english. 
d. commoncrawl_notes.xlxs has my detailed review, but only 4/25 of them seem good enough to actually include in the training data. The only website I'd be proud to include in the training data is the penultimate one of the 25, and even then its not english, its in Chinese. The translation seemed compelling, though.

## extract_text

HTML to text conversion (3 points)
a. The function is implemented in scripts/extract_text.py::extract_text_from_html. The test `uv run pytest -k test_extract_text_from_html_bytes` passes.

b. The if __name__ == "__main__" block in scripts/extract_text.py prints the archive record by record. 
The first english record (same as in the previous question) actually displays the menu data with the appropriate heirarchy. Rather than just being lists of text, as the exerpt below shows:
```
Menu
Benefits & Services
Alumni A-Card
Financial Services
BMO alumni UBC MasterCard
TD Home, Auto & Travel Insurance
```
The resiliparse shows a much richer structure to the text
```
• Benefits & Services
    • Alumni A-Card
    • Financial Services
      • BMO alumni UBC MasterCard
      • TD Home, Auto & Travel Insurance
```
which actually reveals a previously invisible heirarchy, which honestly feels like a godsend. Looking through the rest of the data, however, I do still see some unstripped html like <ul>, <li>, <p>, though those are all things that can be find-and-replace'd, so that's not the end of the world. Those also show up in the .wet files so its not exactly a downside of this parser, either. Ultimately, the version I parsed seems better because the information seems richer, but this is using the basic assumption that richer text is better. Maybe simpler text has better performance, depending on the desired task.
## language_identification
A. The test passes, with the caveat / function interfact that my predict only accepts 1 string rather than a list, like the original fasttext interface does.
B. This assumes that 1. the model has seen the language being used in inference, 2. the prompt is in a single language, or is in a combination of languages it has been trained on, and maybe 3. (this one is most speculative) skills might not automatically transfer across languages. For example, if 90% of the math books are in Russian and English, its not a given that the model will be able to discuss math in Spanish or Korean. Specifically with respect to language identification there are languages with character overlap, like Chinese and Japanese Kanji, which can make it more ambiguous, or if I ask in english "What does omelette du fromage mean?". A similar concern is demonstrated in the `if __name__ == "__main__"` block: 
    `print(predict("hey can you please translate this? Le petit chat noir et blanc dort paisiblement sur le vieux tapis rouge du salon."))`
This returns `('fr', 0.829)`, which could prompt an LLM to respond in French, which is the opposite of helpful! How would I mitigate this? The Bitter Lesson makes me think it'll all be OK, but besides that, if I were in charge of New-GPT-1, I'd limit responses to only be in languages the model has seen, gated by the classifier being very confident about the language.
C. The classifier is actually pretty good with well-structured words, and its confidence is appropriately calibrated. High confidence is typically right. The results with low confidence are the structured data. For example:
`195 fetchTimeMs: 270 charset-detected: UTF-8 languages-cld2: {"reliable":true,"text-bytes":11967,"languages":[{"code":"ru","code-iso-639-3":"rus","text-covered":0.98,"score":566.0,"name":"RUSSIAN"}]}`
has a return tuple of `(en, 0.431164413690567)`, because it's not really sentences but structured data like json and html headers. Once I limited the classifier to only see record.responses, total_docs=20153, en_docs=8183, empty_docs=553, 8183/(20153-553) = 42% of the total documents in here are english. Of the 20 I sampled for the assignment, Types the classifier struggles on are understandable ones where the input is genuinely multiple languages: 
```
Transit Bangkok
EnglishThai FrenchChinese
 
  • 主页
  • 曼谷公交
  • 曼谷机场专线
  • 曼谷大眾運輸系統
  • 曼谷昭披耶河公交船
  • 曼谷运河船
  • 曼谷地铁
  • 曼谷简介
  • 旅行贴士
  • 走出曼谷
  • 联系方式

Bangkok Bus Icon

曼谷公交车: Opposite Soi Chain Sawat

Name of Station: Opposite Soi Chain Sawat
Connecting Lines 22, 45, 46, 72, 109, 115, 116, 149, 507, 1-45, 1-77, 3-13, 3-44, 4-39, 4-53



Name in Thai: ตรงข้ามซอยเจียรสวัสดิ์
From Opposite Soi Chain Sawat to:

To Opposite Soi Chain Sawat from:
```
When randomly sampling 20 values (sampled using scripts/manually_classify.py), it got 19/20 correct, and the only one that wasn't correct was the example above where there is no direct answer. The only one one that was unambiguous to me that it wasn't confidenct on was a list of gun colors, which wasn't in full sentences. This got a confidence of about 0.55:
```
Armor Black
Charcoal Green
Coyote Tan
Hard Coat Anodized
Flat Dark Earth
Burnt Bronze
Midnight Bronze
Bazooka Green
Stormtrooper White
Slate
OD Green
Titanium
Graphite Black
```
Another example where it struggled on the language was on english blogs that talk about networking: 
```
# Install UFW (if not already present)
sudo apt install -y ufw

# Set default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing
```
This is funny, because the classifier doesn't have an option to return `bash` as a language, so its stuck with 0.72 confidence english.

If I'm being charitable, the first example above is Chinese, so I would personally grade it correct, but strictly speaking it was correct on 19/20 sampled, and confident on all the ones with properly structured sentences, rather than code or menus. If I were responsible for fitting the confidence threshold, I have a low tolerance for risk. So if I were doing this in production, I would want the threshold to be something like 0.99, because even if you have 1000 users, 10 of your users, on average, would run into issues. In this code, however, I set the warning threshold to 0.90. The confidences observed in the sample corresponded with inputs I'd welcome as valid input, but ultimately a lever I'd pull is I'd try to improve the classifier to properly label the mislabeled stuff.

After looking at the retention curves at various thresholds (figures/retention_curve.png), I noticed that a vast majority of english articles are lost over threshold 0.8. As it turns out, that is due to three major factors: English occurs in most multilingual webpages, many of the words in English come from intellectual property and are not in the English Dictionary (for example, League of Legends character Irelia or Dr. Mundo are not classified as english despite being in an english article.), and many websites default to english, so the lion's share of menu pages that are just ambiguously classified are in english.

## mask_pii

Personally identifiable information (3 points)
a,b,c. Implemented and tests pass (no IPV6, lots of false positives because my phone number pattern matching doesn't only match real phone numbers)

d. Issues that might arise downstream in a language model when these filters are applied are that the model won't know what the explicit structure of an email address, phone number, or ip address is. There's also the harmless non PII stuff like the local router = 192.168.1.1 which get redacted, 555 numbers might get filtered. The first instinct I would have to mitigate this would be to introduce the model to a whitelist of non-specific values that are generic enough to "leak".

e. Below are the only mistakes from the 20 records I sampled. I'd argue the false negative is worse than the false positive because you're better off erring on the side of not leaking PII
False positive: 
Masked: "|||PHONE_NUMBER|||-AsmyODAfapufbDade4IATxGjmbEoCvmrh9jOWTa" (the true value is "1563679051")
False negative:
Masked: "Telefon 0711-47060191  • |||EMAIL_ADDRESS|||"
True: "Telefon 0711-47060191 • infos@wilih.de"
It catches a correct email, but it doesn't filter the phone number because this isn't the USA format.
## harmful_content

Harmful content (6 points)
a/b. Implemented in scripts/harmful_content_detector.py and the tests pass.
c.  The filters are really really really flimsy. "fùck" was rated as sfw and non-toxic, "cÖck" is rated ('non-nsfw', 1.0), ('non-toxic', 1.0), vs "cock" ('nsfw', 1.0), ('toxic', 1.0). There's also the negative case of wanting a language model to generate the word "cock" in some contexts (What are the steps to firing a gun? What is the name of a male chicken?). I'd mitigate that with contextually appropriate filters on the input, rather than just what might be a big regex or another linear method. I would mitigate this simple approach's downsides by creating whitelisted versions word usage. Also, this probably creates fragile models, because if they *do* get exposed to the data we naively filtered out of it, it'll produce out of distribution behavior. We'd prefer it be prepared with a "mature" knowledge of how to handle this content, but we'd suppress any instinct of it generating that kind of content itself.
d. 
NSFW Classifier test
All 20 cases were correctly identified as non-NSFW, but in the HATE classifier, which used a separate sample below there were NSFW things classified there. Manually running those through the NSFW classifier correctly output as NSFW. All the correct predictions had a high confidence, so as a first pass I'd set the confidence threshold to 0.99, and if we need more tokens later, I can always turn the threshold knob down. 0/20 were labelled nsfw

HATE Classifier test:
All 20 cases were correctly identified as non-Toxic. There were no examples that were actually toxic, but there were examples of divisive content. For example, Israel / Palestine conflict and other politically charged rhetoric. The examples were not hateful or toxic, just partisan, which may not be appropriate for a language model. Regarding a confdience threshold, the smallest confidence observed in the sample is 0.96 and the rest were 0.98+, so I'd stay on the safe side and set the threshold to 0.98, with the same rule as earlier, where I could always lower the threshold later, if necessary. 0/20 were labelled toxic
## gopher_quality_filters

Gopher quality filters (3 points)
a. the tests pass, and the code is implemented in scripts/gopher_quality_filters.py
b. My own judgement says there are some websites in the sample, like a birthday website which got rejected in part because many of the "words" are numbers (years and dates), which fail because the thresholds may too strict. I also haven't seen a representative sample of the internet, so that might still be a sensible threshold. The verdicts on the remaining collection of samples is sensible, but I'm wondering if there's a way to salvage more of the content slop, since I bet a lot of the internet is has menus like that, but many of them probably have usable content once you get past the menu slop.

## quality_classifier

Quality classifier (15 points)


## minhash_deduplication

MinHash + LSH document deduplication (8 points)


## filter_data

Filter data for language modeling (6 points)


## inspect_filtered_data

Inspect filtered data (4 points)


## tokenize_data

Tokenize data (2 points)


## train_model

Train model (8 points)


