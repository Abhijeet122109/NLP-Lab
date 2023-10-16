#Assignment no : 2
#Name : Abhijeet Ashok Landage
#Batch : B2
#Roll no : 36
#Title : Natural Language Processing (NLP) using Gensim

text = ["The food is excellent but the service can be better",
        "The food is always delicious and loved the service",
        "The food was mediocre and the service was terrible"]

g_dict = corpora.Dictionary([simple_preprocess(line) for line in text])
g_bow = [g_dict.doc2bow(simple_preprocess(line)) for line in text]

print("Dictionary : ")
for item in g_bow:
    print([[g_dict[id], freq] for id, freq in item])

g_tfidf = models.TfidfModel(g_bow, smartirs='ntc')

print("TF-IDF Vector:")
for item in g_tfidf[g_bow]:
    print([[g_dict[id], np.around(freq, decimals=2)] for id, freq in item])


#OUTPUT::
Dictionary: [['be', 11, 'better', 1], ['but", 1], ['can', 11, l'excellent', 1], ["food", 1], ['is', 1], ["service", 1], ['the', 211 [['Food", 1], ['is', 1], ['service, 1], ["the", 2], ['always', 1], ["and", 1], ['delicious, 1], ["loved", 1]] II food", 1], I service, 1], ["the", 2], ['and', 1], I'mediocre, 1], ["terrible', 1], ['was, 211

1F-10 Vectori

I'be', 0.44], ['better', 8.44], ["but, 0.44], ["can, 8.44], ['excellent, 8.44], 'is', 0.16]]

[['is', 0.2], ['always', 0.55], and, .2], ['delicious, 9.55], ['loved, 0.5511

[['and', 8.15], ['mediocre', 0.4], ['terrible', 8.4], ["was, 0.81]]