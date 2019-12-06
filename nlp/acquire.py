from requests import get
import pandas as pd
import os
from bs4 import BeautifulSoup


def make_dictionary_from_article(url):
    headers = {'User-Agent':'Codeup Data Science Student'}
    response = get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.title.get_text()
    body = soup.find('div', class_='mk-single-content').get_text()
    # dictionary literal
    return {'title': title,
            'body': body
    }


def make_new_request():
    urls = [
        "https://codeup.com/codeups-data-science-career-accelerator-is-here/",
        "https://codeup.com/data-science-myths/",
        "https://codeup.com/data-science-vs-data-analytics-whats-the-difference/",
        "https://codeup.com/10-tips-to-crush-it-at-the-sa-tech-job-fair/",
        "https://codeup.com/competitor-bootcamps-are-closing-is-the-model-in-danger/",
    ]
    
    output = []
    
    for url in urls:
        output.append(make_dictionary_from_article(url))
        
    df = pd.DataFrame(output)
    df.to_csv('./codeup_blog_posts.csv')

    return output


def get_blog_post():
    filename = './codeup_blog_posts.csv'
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        return make_new_request()

get_blog_post()




    # alternative return statement
    # output = {}
    # output['title] = title
    # output['body] == body
    # return output




