import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Create dataset
data = {
    'title': [
        # Fake News (0)
        "Pope Francis Endorses Donald Trump for President",
        "Hillary Clinton Runs Secret Child Trafficking Ring",
        "FBI Agent Found Dead After Exposing Clinton Corruption",
        "NASA Confirms Earth Will Go Dark for 15 Days in November",
        "Obama Signs Executive Order Banning Pledge of Allegiance",
        "KFC Using Genetically Modified Chickens Without Heads",
        "Coca-Cola Recalls Bottles Containing Human Blood",
        "ISIS Leader Converts to Christianity After Miracle",
        "Trump Promises to Build Wall Around California",
        "Clinton Campaign Paid Beyoncé $10M for Endorsement",
        "Aliens Land in Nevada, Endorse Trump",
        "Florida Man Eats Own Face After Taking New Drug",
        "Putin Warns America: Vote Trump or Face Nuclear War",
        "Michelle Obama Pregnant with Third Child",
        "Disney World Closes Permanently Due to Zika Virus",
        "Hillary Clinton Has Parkinson’s Disease, Doctors Say",
        "Trump to Ban All Muslims from Entering U.S.",
        "Secret Time Machine Found in Area 51",
        "Queen Elizabeth Declares Support for Brexit",
        "Clinton Emails Reveal Plan to Start World War III",
        # Real News (1)
        "U.S. Economy Adds 161,000 Jobs in October",
        "Paris Climate Agreement Enters into Force",
        "Syrian Forces Launch Offensive in Aleppo",
        "Federal Reserve Keeps Interest Rates Unchanged",
        "UK Court Rules Brexit Needs Parliamentary Approval",
        "Cuba’s Fidel Castro Dies at 90",
        "South Korea’s President Faces Impeachment Vote",
        "Italy’s Prime Minister Renzi Resigns After Referendum",
        "Colombia Signs Peace Deal with FARC Rebels",
        "NASA’s Cassini Probe Begins Final Saturn Orbit",
        "Germany Bans Burqa in Public Places",
        "China Launches New Aircraft Carrier",
        "U.S. Approves $3.8B Military Aid to Israel",
        "Turkey Detains Thousands After Failed Coup",
        "Brazil’s Dilma Rousseff Impeached by Senate",
        "Panama Papers Leak Exposes Global Tax Evasion",
        "Zika Virus Spreads Rapidly in Latin America",
        "Apple Unveils iPhone 7 Without Headphone Jack",
        "North Korea Conducts Fifth Nuclear Test",
        "Theresa May Becomes UK Prime Minister"
    ],
    'text': [
        # Fake News Excerpts
        "In a shocking statement, Pope Francis declared his support for Donald Trump, saying he is ‘God’s choice’ for America. The Vatican confirmed the endorsement…",
        "Leaked emails reveal Hillary Clinton operates a child trafficking ring from a Washington, D.C. pizza parlor, sources claim…",
        "An FBI agent investigating Clinton’s emails was found dead in an apparent murder-suicide, raising suspicions of a cover-up…",
        "NASA announced that a rare celestial event will plunge Earth into darkness for 15 days starting November 15…",
        "President Obama signed an executive order banning the Pledge of Allegiance in schools, citing it as divisive…",
        "KFC has been using headless, genetically modified chickens to cut costs, a whistleblower revealed…",
        "Coca-Cola recalled thousands of bottles after tests found traces of human blood in their products…",
        "The leader of ISIS, Abu Bakr al-Baghdadi, converted to Christianity after witnessing a divine miracle, sources say…",
        "Donald Trump vowed to build a wall around California to ‘keep liberals out’ if elected…",
        "Hillary Clinton’s campaign paid Beyoncé $10 million for a concert endorsement, leaked documents show…",
        "A UFO landed in Nevada, and extraterrestrials announced their support for Donald Trump’s campaign…",
        "A Florida man was arrested after eating his own face, allegedly under the influence of a new synthetic drug…",
        "Russian President Vladimir Putin warned that a Clinton victory would lead to nuclear war with Russia…",
        "Insiders claim Michelle Obama is pregnant with her third child at age 52, shocking the nation…",
        "Disney World announced its permanent closure due to the Zika virus outbreak in Florida…",
        "Medical experts claim Hillary Clinton is hiding a Parkinson’s diagnosis, based on her public appearances…",
        "Trump’s campaign announced a total ban on Muslims entering the U.S., effective immediately if he wins…",
        "Declassified documents reveal a working time machine was discovered in Areaandropped to the floor, revealing a secret time machine in Area 51…",
        "Queen Elizabeth stunned the world by publicly supporting Brexit, calling it a ‘new dawn’ for Britain…",
        "Clinton’s emails, obtained by WikiLeaks, reveal a secret plan to provoke World War III with Russia…",
        # Real News Excerpts
        "The U.S. economy added 161,000 jobs in October, with unemployment falling to 4.9%, the Labor Department reported…",
        "The Paris climate agreement, signed by 196 countries, officially entered into force on November 4…",
        "Syrian government forces, backed by Russia, launched a major offensive to recapture rebel-held Aleppo…",
        "The Federal Reserve decided to keep interest rates unchanged, citing stable economic growth…",
        "A UK court ruled that Parliament must approve the government’s plan to trigger Brexit…",
        "Fidel Castro, Cuba’s revolutionary leader, died at age 90, state media announced…",
        "South Korean President Park Geun-hye faces an impeachment vote amid a corruption scandal…",
        "Italian Prime Minister Matteo Renzi resigned after losing a constitutional reform referendum…",
        "Colombia’s government and FARC rebels signed a revised peace deal to end a 52-year conflict…",
        "NASA’s Cassini spacecraft began its final orbits around Saturn, preparing for a crash landing…",
        "Germany’s parliament voted to ban the burqa in public places, citing security concerns…",
        "China launched its second aircraft carrier, a step toward expanding its naval power…",
        "The U.S. Congress approved a $3.8 billion military aid package to Israel over 10 years…",
        "Turkey detained thousands of soldiers, judges, and academics following a failed coup attempt…",
        "Brazil’s Senate voted to impeach President Dilma Rousseff for fiscal mismanagement…",
        "The Panama Papers leak exposed offshore accounts used by global elites to evade taxes…",
        "The Zika virus, linked to birth defects, spread rapidly across Latin America, WHO reported…",
        "Apple unveiled the iPhone 7, removing the headphone jack in a controversial move…",
        "North Korea conducted its fifth nuclear test, defying international sanctions…",
        "Theresa May became UK Prime Minister after David Cameron resigned post-Brexit…",
    ],
    'date': [
        # Fake News Dates
        "2016-10-15",
        "2016-11-02",
        "2016-09-20",
        "2016-08-10",
        "2016-07-25",
        "2016-06-30",
        "2016-05-12",
        "2016-04-18",
        "2016-03-22",
        "2016-02-28",
        "2016-01-15",
        "2016-11-05",
        "2016-10-01",
        "2016-09-08",
        "2016-08-14",
        "2016-07-10",
        "2016-06-05",
        "2016-05-01",
        "2016-04-07",
        "2016-03-03",
        # Real News Dates
        "2016-11-04",
        "2016-11-04",
        "2016-10-27",
        "2016-11-02",
        "2016-11-03",
        "2016-11-25",
        "2016-12-09",
        "2016-12-04",
        "2016-11-24",
        "2016-12-01",
        "2016-12-15",
        "2016-04-20",
        "2016-09-14",
        "2016-07-16",
        "2016-08-31",
        "2016-04-03",
        "2016-02-11",
        "2016-09-07",
        "2016-09-09",
        "2016-07-13",
    ],
    'class': [
        # Fake News Labels
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # Real News Labels
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV for use in your project
df.to_csv("fake_real_news_2016.csv", index=False)

# Visualization 1: Class Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='class', data=df)
plt.title('Distribution of Fake (0) and Real (1) News')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['Fake', 'Real'])
plt.show()

# Visualization 2: Word Clouds
fake_text = ' '.join(df[df['class'] == 0]['text'])
real_text = ' '.join(df[df['class'] == 1]['text'])

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
wordcloud_fake = WordCloud(width=800, height=400, background_color='white').generate(fake_text)
plt.imshow(wordcloud_fake, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Fake News')

plt.subplot(1, 2, 2)
wordcloud_real = WordCloud(width=800, height=400, background_color='white').generate(real_text)
plt.imshow(wordcloud_real, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Real News')
plt.show()