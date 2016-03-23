#http://docs.python-guide.org/en/latest/scenarios/scrape/
#https://www.whisky.com/whisky-database/bottle-search/fdb/Bottles.html?tx_datamintsflaschendb_pi4%5BresultListLayout%5D=1&cHash=662d61e52a87859a720519124b58b568#dm_flaschendb_flasche_search_sorting

from lxml import html
import requests
import numpy as np

#site = 'https://www.whisky.com/whisky-database/bottle-search/fdb/Bottles/List.html'
site = 'https://www.whisky.com/whisky-database/bottle-search/fdb/Bottles.html?tx_datamintsflaschendb_pi4[resultListLayout]=0&cHash=768bf4bcd654312db14a806e9cc7dbc2#dm_flaschendb_flasche_search_sorting'

page = requests.get(site)
tree = html.fromstring(page.content)
c1 = tree.xpath('//span[@class="flaschenIdentifier"]/text()')
c2 = tree.xpath('//span[@class="marke"]/text()')
c3 = tree.xpath('//span[@class="alterEtikett"]/text()')
c4 = tree.xpath('//span[@class="brenndatum"]/text()')
c5 = tree.xpath('//span[@class="abfuelldatum"]/text()')
c6 = tree.xpath('//span[@class="namenszuasatz"]/text()')
c7 = tree.xpath('//span[@class="alkoholgehalt"]/text()')
c8 = tree.xpath('//span[@class="flaschengroesse"]/text()')

print c8
