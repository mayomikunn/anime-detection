from anilist_api import get_anime_poster
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import requests


def display_anime_results(results):
    for anime_name, score in results:
        title, image_url = get_anime_poster(anime_name)
        if image_url:
            img_data = requests.get(image_url).content
            img = Image.open(BytesIO(img_data))

            plt.figure(figsize=(3, 4))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"{title}\nSimilarity: {score:.2f}%", fontsize=10)
            plt.tight_layout()
            plt.show()
