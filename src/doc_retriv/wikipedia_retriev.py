import wikipedia

def search_wikipedia(text):
    return wikipedia.search(text)

def get_page_content(title):
    return wikipedia.page(title).content




