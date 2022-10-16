import urllib.request
from PIL import Image
  
urllib.request.urlretrieve(
  'https://www.signingsavvy.com/images/words/alphabet/2/v1.jpg',
   "gfg.png")
  
img = Image.open("gfg.png")
img.show()