from langconv import Converter

a= open('Ren_text_sentiment_train.txt','r').readlines()
b = open('Ren_text_sentiment_test.txt','r').readlines()
c= open('Ren_text_sentiment_train_h.txt','w')
d = open('Ren_text_sentiment_test_h.txt','w')

c_out = ''
d_out = ''
for i in a:
    txt = Converter('zh-hant').convert(i) 
    c_out += txt
for i in b:
    txt = Converter('zh-hant').convert(i) 
    d_out += txt
c.write(c_out)
d.write(d_out)