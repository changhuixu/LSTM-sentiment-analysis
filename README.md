# LSTM-sentiment-analysis
Due to computationly intensive of LSTM method, we only use two LSTM layes in our classifcation model. These two LSTM layes are bidirectional, which include a forwads LSTM and a backwards LSTM. 

Feature extraction was done by reading all training reviews and tokenizing all english words, as well as removing stop words using `nltk` package.

Training in LSTM RNN contains two steps. First, run the neural network going forward. This sets the cell states.
Then, you go backwards computing derivatives. This uses the cell states (what the network knows at a given point in time) to figure out how to change the network's weights. When LSTM updates cell states, we choose to use the default `Adam` optimizer (http://arxiv.org/abs/1412.6980v8), which is a method for Stochastic Optimization. The optimizer minimizes the loss function, which here is the mean square error between expected output and acutal output.


input matrix shape is (number of samples x maxlen)

`number_of_samples` here is 25000 reviews. All reviews are transform into sequences of word vector.

`maxlen` is the max length of each sequence. i.e., if a review has more than `maxlen` words, then this review will be truncated. However, if a review has less than `maxlen` words, then the sequence will pad 0's to make it a regular shape.

`max_features` is the dictionary size. The dictionary was created before data feed into LSTM RNN. Dictionary keys are purified words, dictionary values are the indicies, which is from 2 to 90000. Such that, the most frequent word has lowest index value. For those rarely occurred words, their indicies is large. We can use `max_features` to filter out uncommon words.

First, keeping the `max_features = 20000`, we tested the effect of `maxlen`, which varied from 25 to 200.

maxlen	|	time (s) | train accuracy | test accuracy
---	|	--- | --- | ---
25 | 618 | 0.9757 | 0.7589
50 | 1113 | 0.9876 | 0.8047
75 | 1507 | 0.9882 | 0.8243
100 | 2004 | 0.9813 | 0.8410
125 | 2435 | 0.9774 | 0.8384
150 | 2939 | 0.9725 | 0.8503
175 | 3352 | 0.9819 | 0.8359
200 | 3811 | 0.9831 | 0.8514

![L1_LSTM](https://github.com/changhuixu/LSTM-sentiment-analysis/blob/master/sentence_length.png "")
The length of sentences are right skewed (Q1:67, Median 92, Q3:152). With squence length of 150, about 75% of reviews are covered. 
![L1_LSTM](https://github.com/changhuixu/LSTM-sentiment-analysis/blob/master/length_accuracy.png "")


Second, keeping the `maxlen = 150`, we tested the effect of `max_features`, which varied from 2500 to 50000.

max_features	| train accuracy | test accuracy
---	| --- | ---
250 | 0.7828 | 0.7722
500 | 0.8392 | 0.8328
1500 | 0.8806 | 0.8554
2500 | 0.9119 | 0.8536
5000 | 0.9324 | 0.8553
10000 | 0.9664 | 0.8412
20000 | 0.9725 | 0.8503
30000 | 0.9850 | 0.8489
40000 | 0.9854 | 0.8321
50000 | 0.9843 | 0.8257
60000 | 0.9854 | 0.8470

![L1_LSTM](https://github.com/changhuixu/LSTM-sentiment-analysis/blob/master/feature_accuracy.png "")

It is interesting to notice that the most frequently appeared 2500 english words could largely determine the sentiment of movie reviews very well.
Britain’s Guardian newspaper, in 1986, estimated the size of the average person’s vocabulary as developing from roughly 300 words at two years old, through 5,000 words at five years old, to some 12,000 words at the age of 12.


## Future impovements

Something that could help cut down on extraneous words is pyenchant https://pythonhosted.org/pyenchant/api/enchant.html. Basic idea is to make your input text a list of words, and fix spelling errors (or recorrect words that shouldn't belong).


## Useful Links

http://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/

http://karpathy.github.io/2015/05/21/rnn-effectiveness/
 
https://github.com/dmnelson/sentiment-analysis-imdb

https://github.com/asampat3090/sentiment-dl

https://github.com/wenjiesha/sentiment_lstm
 
http://blog.csdn.net/zouxy09/article/details/8775518/

http://ir.hit.edu.cn/~dytang/
 
https://apaszke.github.io/lstm-explained.html
 
https://github.com/cjhutto/vaderSentiment

http://www.nltk.org/book/

http://deeplearning.net/software/theano/install_windows.html
