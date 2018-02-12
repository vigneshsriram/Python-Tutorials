Files Included:
1)Amazon_review_300 : Contains 300 reviews taken from amazon
This dataset has 3 columns: label, title, review. We’ll use only “review” here

The TF-IDF is calculated.
First a random review is selected and based om that review 10 other documents are selected which are the most likely to be similar to that review.

Also a boolean variable called lemmatized is included which will decide whether the review should be lemmatized or not. By default it will be saved as FALSE
