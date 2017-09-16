
# import libraries 
import requests
from bs4 import BeautifulSoup

def getReviews(movie_id):
    
    reviews=[]  # variable to hold all reviews
    
    page_url="https://www.rottentomatoes.com/m/"+movie_id+"/reviews/"
    
    while page_url!=None:
        page = requests.get(page_url) 
    
        if page.status_code!=200:    # a status code !=200 indicates a failure, exit the loop 
            page_url=None
        else:                       # status_code 200 indicates success.
            
            # insert your code to process page content
            soup = BeautifulSoup(page.content,'html.parser') #bs4
            divs = soup.select("div.review_table div div div.review_area")
            #divs = soup.select("div.row review_table_row")
            for idx,div in enumerate(divs):
                date = None
                description = None
                score = None
                #get date
                p_date = div.find(class_ ="review_date subtle small").get_text()
                #get description
                p_description = div.find(class_ = "the_review").get_text()
                
                #get score
                p_score = div.find(class_ ="small subtle").get_text()
                str(p_score)
                p_score = p_score[-1:]
                reviews.append((p_date,p_description,p_score))
            
            # GET URL OF NEXT PAGE IF EXISTS, AND START A NEW ROUND OF LOOP    
            nextPage = soup.select("div.content div a.btn.btn-xs.btn-primary-rt")
            
            
            # first set page_url is None. Update this value if you find next page
            # second, look for next page. The URL is specified at (4) in the Figure above
            # third, if next page exists, update page_url using the "href" attribute in <a> tag
            
            page_url=None
            if(not(nextPage[3].get('href') =="#")):
                page_url = "https://www.rottentomatoes.com/" + nextPage[3].get('href')
                
            
            
    return reviews


# best practice to test your class
# if your script is exported as a module,
# the following part is ignored
# this is equivalent to main() in Java

if __name__ == "__main__":  
    
    movie_id='the_imitation_game'
    reviews=getReviews(movie_id)
    print(len(reviews)) #this is total number of reviews
    print(reviews)      #this gives the reviews in the form of tuples