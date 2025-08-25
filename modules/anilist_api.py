import requests


def get_anime_poster(anime_title):
    query = '''
    query ($search: String) {
      Media(search: $search, type: ANIME) {
        title {
          romaji
        }
        coverImage {
          large
        }
      }
    }
    '''
    variables = {'search': anime_title}
    url = 'https://graphql.anilist.co'
    response = requests.post(url, json={'query': query, 'variables': variables})

    if response.status_code == 200:
        data = response.json()
        title = data['data']['Media']['title']['romaji']
        image_url = data['data']['Media']['coverImage']['large']
        return title, image_url
    else:
        print(f"Failed to fetch poster for {anime_title}")
        return anime_title, None
