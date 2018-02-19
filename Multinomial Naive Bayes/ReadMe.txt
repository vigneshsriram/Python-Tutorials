File included here:

1) amazon_review_300.csv : contains 300 reviews from amazon. The columns here are labels, title and text. 

2) amazon_review_large.csv : like the first file it contains 20000 reviews from amazon.

Here we use MultinomialNB to classify the reviews into 2 lables that is 1 and 2.
The tf-idf matrix is generated using 2 different approaches:
- Using TfidfVectorizer from sklearn package
- With lemmatization option 

Then we compare the performances of both the approaches.



Now we want to see how many samples are enough for getting the best classification. This is very important because including more data can lead to drop in the performance and less can give wrong values. 

The best way to solve this problem is to plot a graph with performance on one axis and number of sample on the other.There reaches a point where we can see that the performance stops to increase/stays constant and this is the point where we will get the ideal number of samples need for the Classifier.