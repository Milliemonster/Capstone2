from google_images_download import google_images_download   #importing the library

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"ladybug insect","limit":1000, "chromedriver":'../chromedriver'}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)