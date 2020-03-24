import requests
from bs4 import BeautifulSoup


def extract_news(parser):
    """ Extract news from a given web page """
    news_list = []
    tds = parser.find_all('td', {'class':'title'})
    tds = tds[1:60:2]
    print(len(tds))
    print(tds[0])
    print(tds[27])
    tds_other = parser.find_all('td', {'class':'subtext'})
    new = {}
    for i in range(len(tds_other)):
        #print(tds[i])
        link = tds[i].find('a', {'class':'storylink'}).get('href')
        title = tds[i].find('a', {'class':'storylink'}).contents[0]
        score = tds_other[i].find('span', {'class':'score'}).contents[0]
        author = tds_other[i].find('a', {'class':'hnuser'}).contents[0]
        com = tds_other[i].find_all('a')[5].contents[0].split('\xa0')[0]
        new['link'] = link
        new['title'] = title
        new['score'] = score
        new['author'] = author
        new['com'] = com
        news_list.append(new)
    return news_list


def extract_next_page(parser):
    """ Extract next page URL """
    next_page = parser.find('a', {'class':'morelink'}).get('href')
    return next_page


def get_news(url, n_pages=1):
    """ Collect news from a given web page """
    news = []
    while n_pages:
        print("Collecting data from page: {}".format(url))
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        news_list = extract_news(soup)
        next_page = extract_next_page(soup)
        url = "https://news.ycombinator.com/" + next_page
        news.extend(news_list)
        n_pages -= 1
    return news

